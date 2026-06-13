from __future__ import annotations

import json
import multiprocessing as mp
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dura.data_utils.audit import (
    check_overlap,
    domain_statistics,
    save_audit_report,
    split_statistics,
)
from dura.data_utils.cleaning import CATEGORY_MAP
from dura.data_utils.sampler import build_sampler
from dura.data_utils.splitter import SplitResult, build_splits
from dura.common.hashing import sha256_obj
from dura.common.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl

from dura.config import DURAConfig


@dataclass
class DuraNewsRecord:
    uid: str
    raw_id: str
    label: int
    domain_raw: str
    domain: str
    text: str
    timestamp: str
    comments: str
    image_paths: List[str]
    image_path: str
    has_image: int
    source_split: str
    row_hash: str


@dataclass
class CleaningSummary:
    total_raw: int
    total_clean: int
    removed_duplicates: int
    removed_empty_text: int
    missing_image_count: int
    domain_counts: Dict[str, int]
    label_counts: Dict[str, int]
    duplicate_policy: str


@dataclass
class DataBundle:
    records_by_uid: Dict[str, DuraNewsRecord]
    split: SplitResult
    train_loader: object
    val_loader: object
    test_loader: object
    source_domains: List[str]


def _shared_snapshot_root(cfg: DURAConfig) -> Path:
    raw = str(getattr(cfg.paths, "clean_snapshot_root", "") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            # Keep a shared cache near the configured output root instead of per-run output dirs.
            p = Path(cfg.paths.output_root) / p.name
    else:
        p = Path(cfg.paths.output_root) / "data_snapshots_dura"
    return ensure_dir(p)


def _weibo21_prepare_identity(cfg: DURAConfig) -> Dict[str, object]:
    return {
        "dataset_name": str(cfg.data.dataset_name),
        "dataset_root": str(Path(cfg.paths.dataset_root).resolve()),
        "fake_json": str(cfg.data.fake_json),
        "real_json": str(cfg.data.real_json),
        "image_dirs": list(cfg.data.image_dirs),
        "deduplicate_by_raw_id": bool(cfg.data.deduplicate_by_raw_id),
        "duplicate_policy": str(cfg.data.duplicate_policy),
        "include_comments": bool(cfg.data.include_comments),
        "max_text_chars": int(cfg.data.max_text_chars),
        "max_images_per_post": int(cfg.data.max_images_per_post),
    }


def _weibo21_prepare_cache_dir(cfg: DURAConfig) -> Path:
    ident = _weibo21_prepare_identity(cfg)
    key = sha256_obj(ident)[:24]
    cache_dir = ensure_dir(_shared_snapshot_root(cfg) / f"weibo21_clean_{key}")
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        write_json(meta_path, ident)
    return cache_dir


def _load_cached_weibo21_records(cache_dir: Path) -> tuple[List[DuraNewsRecord], CleaningSummary]:
    rows = read_jsonl(cache_dir / "clean_records.jsonl")
    summary_obj = read_json(cache_dir / "cleaning_summary.json")
    records = [DuraNewsRecord(**row) for row in rows]
    summary = CleaningSummary(**summary_obj)
    return records, summary


def _prepare_weibo21_records(
    cfg: DURAConfig,
    logger,
) -> tuple[List[DuraNewsRecord], CleaningSummary, Path]:
    dataset_root = Path(cfg.paths.dataset_root)
    cache_dir = _weibo21_prepare_cache_dir(cfg)
    records_path = cache_dir / "clean_records.jsonl"
    summary_path = cache_dir / "cleaning_summary.json"
    if records_path.exists() and summary_path.exists():
        records, summary = _load_cached_weibo21_records(cache_dir)
        logger.info(
            "Reused DURA cleaned snapshot | path=%s raw=%s clean=%s missing_image=%s",
            cache_dir,
            summary.total_raw,
            summary.total_clean,
            summary.missing_image_count,
        )
        return records, summary, cache_dir

    logger.info("Preparing DURA cleaned snapshot | path=%s", cache_dir)
    records, summary, _ = load_and_clean_records_multi(
        dataset_root=dataset_root,
        fake_json_rel=cfg.data.fake_json,
        real_json_rel=cfg.data.real_json,
        image_dirs_rel=cfg.data.image_dirs,
        deduplicate_by_raw_id=cfg.data.deduplicate_by_raw_id,
        duplicate_policy=cfg.data.duplicate_policy,
        include_comments=cfg.data.include_comments,
        max_text_chars=cfg.data.max_text_chars,
        max_images_per_post=cfg.data.max_images_per_post,
        snapshot_root=cache_dir,
    )
    return records, summary, cache_dir


def _gossipcop_prepare_identity(cfg: DURAConfig) -> Dict[str, object]:
    return {
        "dataset_name": str(cfg.data.dataset_name),
        "dataset_root": str(Path(cfg.paths.dataset_root).resolve()),
        "gossipcop_real_dir": str(cfg.data.gossipcop_real_dir),
        "gossipcop_fake_dir": str(cfg.data.gossipcop_fake_dir),
        "gossipcop_news_json": str(cfg.data.gossipcop_news_json),
        "gossipcop_images_dirname": str(cfg.data.gossipcop_images_dirname),
        "max_text_chars": int(cfg.data.max_text_chars),
        "max_images_per_post": int(cfg.data.max_images_per_post),
    }


def _gossipcop_prepare_cache_dir(cfg: DURAConfig) -> Path:
    ident = _gossipcop_prepare_identity(cfg)
    key = sha256_obj(ident)[:24]
    cache_dir = ensure_dir(_shared_snapshot_root(cfg) / f"gossipcop_clean_{key}")
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        write_json(meta_path, ident)
    return cache_dir


def _load_cached_gossipcop_records(cache_dir: Path) -> tuple[List[DuraNewsRecord], CleaningSummary]:
    rows = read_jsonl(cache_dir / "clean_records.jsonl")
    summary_obj = read_json(cache_dir / "cleaning_summary.json")
    summary_fields = {f.name for f in CleaningSummary.__dataclass_fields__.values()}
    records = [DuraNewsRecord(**row) for row in rows]
    summary = CleaningSummary(**{k: v for k, v in summary_obj.items() if k in summary_fields})
    return records, summary


def _prepare_gossipcop_records(
    cfg: DURAConfig,
    logger,
) -> tuple[List[DuraNewsRecord], CleaningSummary, Path]:
    cache_dir = _gossipcop_prepare_cache_dir(cfg)
    records_path = cache_dir / "clean_records.jsonl"
    summary_path = cache_dir / "cleaning_summary.json"
    if records_path.exists() and summary_path.exists():
        records, summary = _load_cached_gossipcop_records(cache_dir)
        logger.info(
            "Reused DURA GossipCop snapshot | path=%s raw=%s clean=%s missing_image=%s",
            cache_dir,
            summary.total_raw,
            summary.total_clean,
            summary.missing_image_count,
        )
        return records, summary, cache_dir

    logger.info("Preparing DURA GossipCop snapshot | path=%s", cache_dir)
    records, summary, _ = load_and_clean_gossipcop_records(
        cfg=cfg,
        snapshot_root=cache_dir,
    )
    return records, summary, cache_dir


def _normalize_text(x: str) -> str:
    return " ".join((x or "").replace("\n", " ").split())


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _looks_like_valid_filename(name: str) -> bool:
    if not name:
        return False
    if len(name) > 180:
        return False
    lowered = name.lower()
    return lowered.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))


def _extract_image_candidates(piclists) -> List[str]:
    out: List[str] = []
    if isinstance(piclists, list):
        seq = piclists
    elif isinstance(piclists, str):
        seq = [piclists]
    else:
        seq = []
    for raw in seq:
        s = str(raw or "").strip()
        if not s:
            continue
        try:
            name = Path(s).name.strip()
        except Exception:  # noqa: BLE001
            name = ""
        if _looks_like_valid_filename(name):
            out.append(name)
    return out


def _resolve_image_list(
    raw_id: str,
    image_candidates: List[str],
    image_search_dirs: List[Path],
    max_images_per_post: int,
) -> List[str]:
    found: List[str] = []
    seen: set[str] = set()

    def _append_if_exists(name: str) -> None:
        nonlocal found
        if len(found) >= int(max_images_per_post):
            return
        for d in image_search_dirs:
            p = d / name
            if _safe_exists(p):
                sp = str(p)
                if sp not in seen:
                    seen.add(sp)
                    found.append(sp)
                return

    for name in image_candidates:
        _append_if_exists(name)

    # Fallback: collect all numeric suffix image variants in deterministic order.
    for d in image_search_dirs:
        for suffix in ["jpg", "jpeg", "png", "bmp", "gif", "webp"]:
            for idx in range(0, 32):
                if len(found) >= int(max_images_per_post):
                    break
                p = d / f"{raw_id}_{idx}.{suffix}"
                if _safe_exists(p):
                    sp = str(p)
                    if sp not in seen:
                        seen.add(sp)
                        found.append(sp)
            if len(found) >= int(max_images_per_post):
                break
        if len(found) >= int(max_images_per_post):
            break

    return found


def _parse_weibo_image_candidates(raw: str) -> List[str]:
    out: List[str] = []
    for token in (raw or "").strip().split("|"):
        t = token.strip()
        if (not t) or t.lower() == "null":
            continue
        name = Path(t.split("?", 1)[0]).name.strip()
        if name:
            out.append(name)
    return out


def _resolve_weibo_image_list(
    candidates: List[str],
    label: int,
    rumor_dir: Path,
    nonrumor_dir: Path,
    max_images_per_post: int,
) -> List[str]:
    if not candidates:
        return []
    preferred = [rumor_dir, nonrumor_dir] if int(label) == 1 else [nonrumor_dir, rumor_dir]
    found: List[str] = []
    seen: set[str] = set()
    for name in candidates:
        for base in preferred:
            p = base / name
            if _safe_exists(p):
                sp = str(p)
                if sp not in seen:
                    seen.add(sp)
                    found.append(sp)
                break
        if len(found) >= int(max_images_per_post):
            break
    return found


def _iter_weibo_triplets(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        while True:
            meta = f.readline()
            if not meta:
                break
            image_line = f.readline()
            text_line = f.readline()
            if (not image_line) or (not text_line):
                break
            yield meta.rstrip("\n"), image_line.rstrip("\n"), text_line.rstrip("\n")


def _read_split_pickle(path: Path) -> Dict[str, int]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid split pickle (expect dict): {path}")
    out: Dict[str, int] = {}
    for k, v in obj.items():
        out[str(k)] = int(v)
    return out


def _apply_weibo_domain_map(records_by_uid: Dict[str, DuraNewsRecord], domain_map_path: Path) -> int:
    obj = read_json(domain_map_path)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid weibo domain map json: {domain_map_path}")

    if "uid_map" in obj and isinstance(obj["uid_map"], dict):
        uid_map = obj["uid_map"]
    elif "rows" in obj and isinstance(obj["rows"], list):
        uid_map = {str(row["uid"]): row for row in obj["rows"] if isinstance(row, dict) and "uid" in row}
    else:
        uid_map = obj

    applied = 0
    for uid, payload in uid_map.items():
        if uid not in records_by_uid:
            continue
        if isinstance(payload, str):
            domain = payload
            domain_raw = payload
        elif isinstance(payload, dict):
            domain = str(payload.get("domain", payload.get("category", ""))).strip()
            domain_raw = str(payload.get("domain_raw", payload.get("category_raw", domain))).strip() or domain
        else:
            continue
        if not domain:
            continue
        records_by_uid[uid].domain = domain
        records_by_uid[uid].domain_raw = domain_raw
        applied += 1
    return applied


def _load_json_dict(path: Path) -> Optional[dict]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return None
    if isinstance(obj, dict):
        return obj
    return None


def _json_as_text(meta: dict, key: str) -> str:
    value = meta.get(key, "")
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(str(v) for v in value if v is not None)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _build_gossipcop_text(meta: dict, max_text_chars: int) -> str:
    title = _normalize_text(_json_as_text(meta, "title"))
    text = _normalize_text(_json_as_text(meta, "text"))
    if title and text:
        merged = text if text.startswith(title) else f"{title} {text}"
    else:
        merged = title or text
    return _normalize_text(merged[:max_text_chars])


def _pick_local_gossipcop_images(image_dir: Path, max_images_per_post: int) -> List[str]:
    if (not _safe_exists(image_dir)) or (not image_dir.is_dir()):
        return []
    found: List[str] = []
    for p in sorted(image_dir.iterdir()):
        if len(found) >= int(max_images_per_post):
            break
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".svg"}:
            found.append(str(p))
    return found


def _resolve_gossipcop_image_list(
    sample_dir: Path,
    meta: dict,
    images_dirname: str,
    max_images_per_post: int,
) -> List[str]:
    found = _pick_local_gossipcop_images(sample_dir / images_dirname, max_images_per_post=max_images_per_post)
    if found:
        return found

    images = meta.get("images")
    if not isinstance(images, list):
        return []

    resolved: List[str] = []
    seen: set[str] = set()
    for item in images:
        if len(resolved) >= int(max_images_per_post):
            break
        if not isinstance(item, str):
            continue
        raw = item.strip()
        if not raw:
            continue
        p = Path(raw)
        candidates = []
        if _safe_exists(p):
            candidates.append(p)
        name = p.name.strip()
        if name:
            candidates.append(sample_dir / images_dirname / name)
        for candidate in candidates:
            if _safe_exists(candidate) and candidate.is_file():
                sc = str(candidate)
                if sc not in seen:
                    seen.add(sc)
                    resolved.append(sc)
                break
    return resolved


def load_and_clean_gossipcop_records(
    cfg: DURAConfig,
    snapshot_root: Path,
) -> tuple[List[DuraNewsRecord], CleaningSummary, Path]:
    root = Path(cfg.paths.dataset_root)
    class_specs = [
        (cfg.data.gossipcop_real_dir, 0, "real"),
        (cfg.data.gossipcop_fake_dir, 1, "fake"),
    ]

    raw_count = 0
    removed_empty_text = 0
    removed_missing_json = 0
    removed_bad_json = 0
    removed_duplicates = 0
    label_counts = {"0": 0, "1": 0}
    domain_counts: Dict[str, int] = {}
    records_by_uid: Dict[str, DuraNewsRecord] = {}

    for rel_dir, label, src in class_specs:
        class_dir = root / rel_dir
        if not _safe_exists(class_dir):
            raise RuntimeError(f"Missing gossipcop class dir: {class_dir}")

        for sample_dir in sorted(class_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            raw_count += 1
            raw_id = sample_dir.name.strip()
            if not raw_id:
                continue
            if raw_id in records_by_uid:
                removed_duplicates += 1
                continue

            json_path = sample_dir / cfg.data.gossipcop_news_json
            if not _safe_exists(json_path):
                removed_missing_json += 1
                continue

            meta = _load_json_dict(json_path)
            if meta is None:
                removed_bad_json += 1
                continue

            text_norm = _build_gossipcop_text(meta, max_text_chars=cfg.data.max_text_chars)
            if not text_norm:
                removed_empty_text += 1
                continue

            image_paths = _resolve_gossipcop_image_list(
                sample_dir=sample_dir,
                meta=meta,
                images_dirname=cfg.data.gossipcop_images_dirname,
                max_images_per_post=cfg.data.max_images_per_post,
            )
            image_path = image_paths[0] if image_paths else ""
            has_image = int(bool(image_paths))
            domain = "gossipcop"
            row_hash = sha256_obj(
                {
                    "id": raw_id,
                    "label": int(label),
                    "text": text_norm,
                    "image_paths": list(image_paths),
                }
            )
            records_by_uid[raw_id] = DuraNewsRecord(
                uid=raw_id,
                raw_id=raw_id,
                label=int(label),
                domain_raw=domain,
                domain=domain,
                text=text_norm,
                timestamp="",
                comments="",
                image_paths=list(image_paths),
                image_path=image_path,
                has_image=has_image,
                source_split=src,
                row_hash=row_hash,
            )
            label_counts[str(int(label))] += 1
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

    cleaned = [records_by_uid[uid] for uid in sorted(records_by_uid.keys())]
    missing_image_count = sum(1 for r in cleaned if r.has_image == 0)
    summary = CleaningSummary(
        total_raw=raw_count,
        total_clean=len(cleaned),
        removed_duplicates=removed_duplicates,
        removed_empty_text=removed_empty_text,
        missing_image_count=missing_image_count,
        domain_counts=dict(sorted(domain_counts.items())),
        label_counts=label_counts,
        duplicate_policy="keep_first",
    )

    snapshot_dir = ensure_dir(snapshot_root)
    write_jsonl(snapshot_dir / "clean_records.jsonl", [asdict(r) for r in cleaned])
    write_json(
        snapshot_dir / "cleaning_summary.json",
        {
            **asdict(summary),
            "removed_missing_json": removed_missing_json,
            "removed_bad_json": removed_bad_json,
            "removed_empty_text_only": removed_empty_text,
        },
    )
    return cleaned, summary, snapshot_dir


def load_and_clean_records_multi(
    dataset_root: Path,
    fake_json_rel: str,
    real_json_rel: str,
    image_dirs_rel: List[str],
    deduplicate_by_raw_id: bool,
    duplicate_policy: str,
    include_comments: bool,
    max_text_chars: int,
    max_images_per_post: int,
    snapshot_root: Path,
) -> tuple[List[DuraNewsRecord], CleaningSummary, Path]:
    fake_path = dataset_root / fake_json_rel
    real_path = dataset_root / real_json_rel
    image_dirs = [dataset_root / p for p in image_dirs_rel]

    fake_rows = read_jsonl(fake_path)
    real_rows = read_jsonl(real_path)
    all_rows = [("fake", row) for row in fake_rows] + [("real", row) for row in real_rows]

    cleaned: List[DuraNewsRecord] = []
    seen_idx: Dict[str, int] = {}
    removed_duplicates = 0
    removed_empty_text = 0
    uid_counters: Dict[str, int] = {}

    for src_split, row in all_rows:
        raw_id = str(row.get("id", "")).strip()
        if not raw_id:
            continue

        text = _normalize_text(str(row.get("content", "")))
        comments = _normalize_text(str(row.get("comments", "")))
        if include_comments and comments:
            text = f"{text} [COMMENTS] {comments}".strip()
        if not text:
            removed_empty_text += 1
            continue
        text = text[:max_text_chars]

        label = int(row.get("label", 1 if src_split == "fake" else 0))
        domain_raw = str(row.get("category", "未知")).strip() or "未知"
        domain = CATEGORY_MAP.get(domain_raw, domain_raw)

        image_candidates = _extract_image_candidates(row.get("piclists", None))
        image_paths = _resolve_image_list(
            raw_id=raw_id,
            image_candidates=image_candidates,
            image_search_dirs=image_dirs,
            max_images_per_post=max_images_per_post,
        )
        image_path = image_paths[0] if image_paths else ""
        has_image = int(len(image_paths) > 0)

        payload_for_hash = {
            "id": raw_id,
            "label": label,
            "domain": domain,
            "text": text,
            "image_paths": list(image_paths),
        }

        uid_key = f"{raw_id}#{src_split}"
        uid_idx = uid_counters.get(uid_key, 0)
        uid_counters[uid_key] = uid_idx + 1
        uid = uid_key if uid_idx == 0 else f"{uid_key}#{uid_idx}"

        record = DuraNewsRecord(
            uid=uid,
            raw_id=raw_id,
            label=label,
            domain_raw=domain_raw,
            domain=domain,
            text=text,
            timestamp=str(row.get("timestamp", "")),
            comments=comments,
            image_paths=list(image_paths),
            image_path=image_path,
            has_image=has_image,
            source_split=src_split,
            row_hash=sha256_obj(payload_for_hash),
        )

        if deduplicate_by_raw_id:
            if raw_id in seen_idx:
                removed_duplicates += 1
                if duplicate_policy == "keep_last":
                    cleaned[seen_idx[raw_id]] = record
                continue
            seen_idx[raw_id] = len(cleaned)

        cleaned.append(record)

    missing_image_count = sum(1 for r in cleaned if r.has_image == 0)
    domain_counts: Dict[str, int] = {}
    label_counts: Dict[str, int] = {"0": 0, "1": 0}
    for r in cleaned:
        domain_counts[r.domain] = domain_counts.get(r.domain, 0) + 1
        label_counts[str(r.label)] = label_counts.get(str(r.label), 0) + 1

    summary = CleaningSummary(
        total_raw=len(all_rows),
        total_clean=len(cleaned),
        removed_duplicates=removed_duplicates,
        removed_empty_text=removed_empty_text,
        missing_image_count=missing_image_count,
        domain_counts=dict(sorted(domain_counts.items())),
        label_counts=label_counts,
        duplicate_policy=duplicate_policy,
    )

    snapshot_dir = ensure_dir(snapshot_root)
    write_jsonl(snapshot_dir / "clean_records.jsonl", [asdict(r) for r in cleaned])
    write_json(snapshot_dir / "cleaning_summary.json", asdict(summary))
    return cleaned, summary, snapshot_dir


def load_weibo_records_and_split(
    cfg: DURAConfig,
    run_dir: Path,
) -> tuple[Dict[str, DuraNewsRecord], SplitResult, Dict[str, object]]:
    root = Path(cfg.paths.dataset_root)
    tweets_dir = root / cfg.data.weibo_tweets_dir
    rumor_dir = root / cfg.data.weibo_rumor_image_dir
    nonrumor_dir = root / cfg.data.weibo_nonrumor_image_dir

    file_specs = [
        (cfg.data.weibo_train_nonrumor_file, 0, "train_nonrumor"),
        (cfg.data.weibo_test_nonrumor_file, 0, "test_nonrumor"),
        (cfg.data.weibo_train_rumor_file, 1, "train_rumor"),
        (cfg.data.weibo_test_rumor_file, 1, "test_rumor"),
    ]

    raw_count = 0
    removed_empty_text = 0
    duplicate_ids = 0
    records_by_uid: Dict[str, DuraNewsRecord] = {}

    for rel_file, label, src in file_specs:
        fpath = tweets_dir / rel_file
        if not _safe_exists(fpath):
            raise RuntimeError(f"Missing weibo tweet file: {fpath}")

        for meta, images, text in _iter_weibo_triplets(fpath):
            raw_count += 1
            raw_id = meta.split("|", 1)[0].strip()
            if not raw_id:
                continue

            text_norm = _normalize_text(text)[: cfg.data.max_text_chars]
            if not text_norm:
                removed_empty_text += 1
                continue
            if raw_id in records_by_uid:
                duplicate_ids += 1
                continue

            image_paths = _resolve_weibo_image_list(
                candidates=_parse_weibo_image_candidates(images),
                label=label,
                rumor_dir=rumor_dir,
                nonrumor_dir=nonrumor_dir,
                max_images_per_post=cfg.data.max_images_per_post,
            )
            image_path = image_paths[0] if image_paths else ""
            has_image = int(bool(image_paths))
            row_hash = sha256_obj(
                {
                    "id": raw_id,
                    "label": int(label),
                    "text": text_norm,
                    "image_paths": list(image_paths),
                }
            )
            records_by_uid[raw_id] = DuraNewsRecord(
                uid=raw_id,
                raw_id=raw_id,
                label=int(label),
                domain_raw="unknown",
                domain="unknown",
                text=text_norm,
                timestamp="",
                comments="",
                image_paths=list(image_paths),
                image_path=image_path,
                has_image=has_image,
                source_split=src,
                row_hash=row_hash,
            )

    train_map = _read_split_pickle(root / cfg.data.weibo_train_split_pickle)
    val_map = _read_split_pickle(root / cfg.data.weibo_val_split_pickle)
    test_map = _read_split_pickle(root / cfg.data.weibo_test_split_pickle)

    train_ids = [uid for uid in train_map.keys() if uid in records_by_uid]
    val_ids = [uid for uid in val_map.keys() if uid in records_by_uid]
    test_ids = [uid for uid in test_map.keys() if uid in records_by_uid]

    missing_split_ids = {
        "train": len(train_map) - len(train_ids),
        "val": len(val_map) - len(val_ids),
        "test": len(test_map) - len(test_ids),
    }

    event_map: Dict[str, int] = {}
    event_map.update(train_map)
    event_map.update(val_map)
    event_map.update(test_map)
    for uid, event_id in event_map.items():
        if uid not in records_by_uid:
            continue
        domain = f"event_{int(event_id)}"
        records_by_uid[uid].domain = domain
        records_by_uid[uid].domain_raw = domain

    split_map_path = Path(cfg.paths.split_map_path) if cfg.paths.split_map_path else run_dir / "split_map.json"
    split_payload = {
        "mode": "weibo_predefined",
        "dataset_name": "weibo",
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }
    write_json(split_map_path, split_payload)
    split = SplitResult(
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        test_ids=sorted(test_ids),
        split_map_path=split_map_path,
    )

    if cfg.data.weibo_domain_map_json:
        applied = _apply_weibo_domain_map(records_by_uid, Path(cfg.data.weibo_domain_map_json))
    else:
        applied = 0

    selected = set(split.train_ids + split.val_ids + split.test_ids)
    records_by_uid = {uid: r for uid, r in records_by_uid.items() if uid in selected}

    snapshot_root = ensure_dir(run_dir / "data_snapshot")
    write_jsonl(snapshot_root / "clean_records.jsonl", [asdict(records_by_uid[uid]) for uid in sorted(records_by_uid.keys())])
    missing_image_count = sum(1 for r in records_by_uid.values() if r.has_image == 0)
    summary = {
        "total_raw": raw_count,
        "total_clean": len(records_by_uid),
        "removed_empty_text": removed_empty_text,
        "removed_duplicates": duplicate_ids,
        "missing_image_count": missing_image_count,
        "missing_split_ids": missing_split_ids,
        "domain_labels_applied": applied,
    }
    return records_by_uid, split, summary


class DuraMultiImageDataset(Dataset):
    def __init__(
        self,
        records_by_uid: Dict[str, DuraNewsRecord],
        uids: List[str],
        image_size: int,
        max_images_per_post: int,
    ) -> None:
        self.records_by_uid = records_by_uid
        self.uids = list(uids)
        self.image_size = int(image_size)
        self.max_images_per_post = int(max_images_per_post)

    def __len__(self) -> int:
        return len(self.uids)

    def _load_image_tensor(self, image_path: str) -> tuple[torch.Tensor, float]:
        if not image_path:
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32), 0.0
        p = Path(image_path)
        if not p.exists():
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32), 0.0
        try:
            img = Image.open(p).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            arr = np.asarray(img).astype("float32") / 255.0
            arr = np.transpose(arr, (2, 0, 1))
            x = torch.from_numpy(arr)
            return x, 1.0
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32), 0.0

    def __getitem__(self, idx: int) -> Dict[str, object]:
        uid = self.uids[idx]
        r = self.records_by_uid[uid]
        images: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        chosen = list(r.image_paths[: self.max_images_per_post])
        for image_path in chosen:
            x, m = self._load_image_tensor(image_path)
            images.append(x)
            masks.append(torch.tensor(m, dtype=torch.float32))
        while len(images) < self.max_images_per_post:
            images.append(torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32))
            masks.append(torch.tensor(0.0, dtype=torch.float32))
        return {
            "uid": uid,
            "raw_id": r.raw_id,
            "domain": r.domain,
            "text": r.text,
            "images": torch.stack(images, dim=0),
            "image_mask": torch.stack(masks, dim=0),
            "image_paths": list(chosen),
            "label": torch.tensor(r.label, dtype=torch.long),
        }


class _FeatureCacheDataset(Dataset):
    def __init__(
        self,
        records_by_uid: Dict[str, DuraNewsRecord],
        uids: List[str],
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> None:
        self.records_by_uid = records_by_uid
        self.uids = list(uids)
        self.text_features = text_features
        self.image_features = image_features
        self.image_mask = image_mask
        self._uid_to_idx = {uid: i for i, uid in enumerate(self.uids)}

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        uid = self.uids[idx]
        r = self.records_by_uid[uid]
        feat_idx = self._uid_to_idx[uid]
        return {
            "uid": uid,
            "raw_id": r.raw_id,
            "domain": r.domain,
            "text": r.text,
            "text_feature": self.text_features[feat_idx],
            "image_features": self.image_features[feat_idx],
            "image_mask": self.image_mask[feat_idx],
            "image_paths": list(r.image_paths),
            "label": torch.tensor(r.label, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "uid": [x["uid"] for x in batch],
        "raw_id": [x["raw_id"] for x in batch],
        "domain": [x["domain"] for x in batch],
        "text": [x["text"] for x in batch],
        "images": torch.stack([x["images"] for x in batch], dim=0),
        "image_mask": torch.stack([x["image_mask"] for x in batch], dim=0),
        "image_paths": [x["image_paths"] for x in batch],
        "label": torch.stack([x["label"] for x in batch], dim=0),
    }


def feature_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "uid": [x["uid"] for x in batch],
        "raw_id": [x["raw_id"] for x in batch],
        "domain": [x["domain"] for x in batch],
        "text": [x["text"] for x in batch],
        "text_feature": torch.stack([x["text_feature"] for x in batch], dim=0),
        "image_features": torch.stack([x["image_features"] for x in batch], dim=0),
        "image_mask": torch.stack([x["image_mask"] for x in batch], dim=0),
        "image_paths": [x["image_paths"] for x in batch],
        "label": torch.stack([x["label"] for x in batch], dim=0),
    }


def _can_use_mp_workers() -> tuple[bool, str]:
    try:
        ctx = mp.get_context("spawn")
        q = ctx.Queue(maxsize=1)
        q.close()
        q.join_thread()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _resolve_loader_runtime(cfg: DURAConfig, logger) -> None:
    if getattr(cfg, "_loader_runtime_resolved", False):
        return
    effective_workers = int(cfg.data.num_workers)
    effective_persistent = bool(cfg.data.persistent_workers and effective_workers > 0)
    if effective_workers > 0:
        ok, reason = _can_use_mp_workers()
        if not ok:
            logger.warning("DataLoader multi-worker unavailable, fallback to num_workers=0 | reason=%s", reason)
            effective_workers = 0
            effective_persistent = False
    cfg._effective_num_workers = effective_workers
    cfg._effective_persistent_workers = effective_persistent
    cfg._loader_runtime_resolved = True


def _slice_feature_cache(feature_cache: Optional[dict], uids: List[str]) -> Optional[dict]:
    if feature_cache is None:
        return None
    uid_to_index = feature_cache["uid_to_index"]
    missing = [uid for uid in uids if uid not in uid_to_index]
    if missing:
        return None
    idx = torch.tensor([uid_to_index[uid] for uid in uids], dtype=torch.long)
    return {
        "text_features": feature_cache["text_features"].index_select(0, idx),
        "image_features": feature_cache["image_features"].index_select(0, idx),
        "image_mask": feature_cache["image_mask"].index_select(0, idx),
    }


def build_loader(
    records_by_uid: Dict[str, DuraNewsRecord],
    uids: List[str],
    batch_size: int,
    image_size: int,
    max_images_per_post: int,
    sampler_strategy: str,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    feature_cache: Optional[dict] = None,
) -> DataLoader:
    if feature_cache is not None:
        dataset = _FeatureCacheDataset(
            records_by_uid=records_by_uid,
            uids=uids,
            text_features=feature_cache["text_features"],
            image_features=feature_cache["image_features"],
            image_mask=feature_cache["image_mask"],
        )
        collate = feature_collate_fn
    else:
        dataset = DuraMultiImageDataset(
            records_by_uid=records_by_uid,
            uids=uids,
            image_size=image_size,
            max_images_per_post=max_images_per_post,
        )
        collate = collate_fn
    sampler = build_sampler(strategy=sampler_strategy, uids=uids, records_by_uid=records_by_uid)
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (sampler is None and shuffle),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def make_loader_for_uids(
    cfg: DURAConfig,
    records_by_uid: Dict[str, DuraNewsRecord],
    uids: List[str],
    is_train: bool,
    feature_cache: Optional[dict] = None,
):
    num_workers = int(getattr(cfg, "_effective_num_workers", cfg.data.num_workers))
    persistent_workers = bool(getattr(cfg, "_effective_persistent_workers", cfg.data.persistent_workers and num_workers > 0))
    feature_cache_slice = _slice_feature_cache(feature_cache, uids)
    pin_memory = bool(cfg.data.pin_memory)
    if feature_cache_slice is not None:
        # Cached features are already resident CPU tensors. Multi-worker loading adds
        # IPC/worker scheduling overhead and can cause bursty stalls without any
        # disk-I/O benefit. Pinning is also unnecessary here because the training
        # path does not currently use non_blocking GPU transfers.
        num_workers = 0
        persistent_workers = False
        pin_memory = False
    return build_loader(
        records_by_uid=records_by_uid,
        uids=uids,
        batch_size=(cfg.data.batch_size if is_train else cfg.data.eval_batch_size),
        image_size=cfg.clip.image_size,
        max_images_per_post=cfg.data.max_images_per_post,
        sampler_strategy=(cfg.data.sampler if is_train else "none"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=bool(is_train),
        persistent_workers=persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        feature_cache=feature_cache_slice,
    )


def create_data_bundle(
    cfg: DURAConfig,
    run_dir: Path,
    logger,
    feature_cache: Optional[dict] = None,
    build_loaders: bool = True,
) -> DataBundle:
    _resolve_loader_runtime(cfg, logger)
    dataset_root = Path(cfg.paths.dataset_root)
    snapshot_root = run_dir / "data_snapshot"

    if cfg.data.dataset_name == "weibo21":
        records, summary, shared_snapshot_dir = _prepare_weibo21_records(cfg=cfg, logger=logger)
        ensure_dir(snapshot_root)
        write_json(
            snapshot_root / "cache_ref.json",
            {
                "shared_snapshot_dir": str(shared_snapshot_dir),
                "clean_records_jsonl": str(shared_snapshot_dir / "clean_records.jsonl"),
                "cleaning_summary_json": str(shared_snapshot_dir / "cleaning_summary.json"),
            },
        )
        logger.info(
            "Loaded DURA records | raw=%s clean=%s duplicates_removed=%s missing_image=%s",
            summary.total_raw,
            summary.total_clean,
            summary.removed_duplicates,
            summary.missing_image_count,
        )

        split_map_path = Path(cfg.paths.split_map_path) if cfg.paths.split_map_path else run_dir / "split_map.json"
        split = build_splits(
            records=records,
            mode=cfg.experiment.mode,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.data.split_seed,
            split_map_path=split_map_path,
        )
        records_by_uid = {r.uid: r for r in records}
        source_domains: Optional[List[str]] = None
    elif cfg.data.dataset_name == "gossipcop":
        records, summary, shared_snapshot_dir = _prepare_gossipcop_records(cfg=cfg, logger=logger)
        ensure_dir(snapshot_root)
        write_json(
            snapshot_root / "cache_ref.json",
            {
                "shared_snapshot_dir": str(shared_snapshot_dir),
                "clean_records_jsonl": str(shared_snapshot_dir / "clean_records.jsonl"),
                "cleaning_summary_json": str(shared_snapshot_dir / "cleaning_summary.json"),
            },
        )
        logger.info(
            "Loaded DURA GossipCop records | raw=%s clean=%s duplicates_removed=%s missing_image=%s",
            summary.total_raw,
            summary.total_clean,
            summary.removed_duplicates,
            summary.missing_image_count,
        )

        split_map_path = Path(cfg.paths.split_map_path) if cfg.paths.split_map_path else run_dir / "split_map.json"
        split = build_splits(
            records=records,
            mode=cfg.experiment.mode,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            seed=cfg.data.split_seed,
            split_map_path=split_map_path,
        )
        records_by_uid = {r.uid: r for r in records}
        source_domains = None
    elif cfg.data.dataset_name == "weibo":
        records_by_uid, split, summary = load_weibo_records_and_split(cfg=cfg, run_dir=run_dir)
        records = list(records_by_uid.values())
        logger.info(
            "Loaded DURA weibo records | raw=%s clean=%s duplicates_removed=%s missing_image=%s",
            summary["total_raw"],
            summary["total_clean"],
            summary["removed_duplicates"],
            summary["missing_image_count"],
        )
        source_domains = sorted({records_by_uid[uid].domain for uid in split.train_ids})
    else:
        raise ValueError(f"Unsupported data.dataset_name for DURA: {cfg.data.dataset_name}")

    missing_train = [uid for uid in split.train_ids if uid not in records_by_uid]
    missing_val = [uid for uid in split.val_ids if uid not in records_by_uid]
    missing_test = [uid for uid in split.test_ids if uid not in records_by_uid]
    if missing_train or missing_val or missing_test:
        logger.warning(
            "Filtered missing split ids | train=%s val=%s test=%s",
            len(missing_train),
            len(missing_val),
            len(missing_test),
        )
        split.train_ids = [uid for uid in split.train_ids if uid in records_by_uid]
        split.val_ids = [uid for uid in split.val_ids if uid in records_by_uid]
        split.test_ids = [uid for uid in split.test_ids if uid in records_by_uid]

    if int(cfg.data.max_train_samples) > 0:
        split.train_ids = split.train_ids[: int(cfg.data.max_train_samples)]
    if int(cfg.data.max_eval_samples) > 0:
        split.val_ids = split.val_ids[: int(cfg.data.max_eval_samples)]
        split.test_ids = split.test_ids[: int(cfg.data.max_eval_samples)]
    if cfg.data.dataset_name == "weibo":
        source_domains = sorted({records_by_uid[uid].domain for uid in split.train_ids})

    overlap = check_overlap(split.train_ids, split.val_ids, split.test_ids)
    split_stats = {
        "train": split_statistics(records_by_uid, split.train_ids),
        "val": split_statistics(records_by_uid, split.val_ids),
        "test": split_statistics(records_by_uid, split.test_ids),
    }
    domain_stats = domain_statistics(records)
    save_audit_report(
        out_dir=ensure_dir(run_dir / "audit"),
        overlap=overlap,
        split_stats=split_stats,
        domain_stats=domain_stats,
    )
    if overlap.has_overlap:
        raise RuntimeError("Data split overlap detected; see audit report")

    if build_loaders:
        train_loader = make_loader_for_uids(cfg, records_by_uid, split.train_ids, is_train=True, feature_cache=feature_cache)
        val_loader = make_loader_for_uids(cfg, records_by_uid, split.val_ids, is_train=False, feature_cache=feature_cache)
        test_loader = make_loader_for_uids(cfg, records_by_uid, split.test_ids, is_train=False, feature_cache=feature_cache)
    else:
        train_loader = None
        val_loader = None
        test_loader = None

    if source_domains is None:
        source_domains = sorted({r.domain for r in records})

    logger.info(
        "DURA split sizes | train=%s val=%s test=%s | mode=%s",
        len(split.train_ids),
        len(split.val_ids),
        len(split.test_ids),
        cfg.experiment.mode,
    )

    return DataBundle(
        records_by_uid=records_by_uid,
        split=split,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        source_domains=source_domains,
    )


def domain_specific_loaders(
    cfg: DURAConfig,
    records_by_uid: Dict[str, DuraNewsRecord],
    train_ids: List[str],
    val_ids: List[str],
    domain: str,
    feature_cache: Optional[dict] = None,
):
    d_train = [uid for uid in train_ids if records_by_uid[uid].domain == domain]
    d_val = [uid for uid in val_ids if records_by_uid[uid].domain == domain]
    if len(d_train) == 0:
        return None, None, []
    if len(d_val) == 0:
        cut = max(1, int(len(d_train) * 0.1))
        d_val = d_train[:cut]
        d_train = d_train[cut:] if len(d_train) > cut else d_train
    train_loader = make_loader_for_uids(cfg, records_by_uid, d_train, is_train=True, feature_cache=feature_cache)
    val_loader = make_loader_for_uids(cfg, records_by_uid, d_val, is_train=False, feature_cache=feature_cache)
    return train_loader, val_loader, d_train
