from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from dura.common.hashing import sha256_obj
from dura.common.io import ensure_dir, read_jsonl, write_json, write_jsonl


CATEGORY_MAP = {
    "社会生活": "社会",
    "文体娱乐": "娱乐",
    "财经商业": "财经",
    "医药健康": "健康",
    "灾难事故": "灾难",
    "教育考试": "教育",
    "科技": "科学",
    "政治": "政治",
    "军事": "军事",
}


@dataclass
class NewsRecord:
    uid: str
    raw_id: str
    label: int
    domain_raw: str
    domain: str
    text: str
    timestamp: str
    comments: str
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



def _normalize_text(x: str) -> str:
    return " ".join((x or "").replace("\n", " ").split())



def _extract_first_image_path(piclists) -> Optional[str]:
    if isinstance(piclists, list):
        if not piclists:
            return None
        first = str(piclists[0]).strip()
        return first or None
    if isinstance(piclists, str):
        s = piclists.strip()
        return s or None
    return None



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
    return lowered.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif"))



def _resolve_image(raw_id: str, img_hint: Optional[str], image_search_dirs: List[Path]) -> Optional[Path]:
    if img_hint:
        try:
            name = Path(img_hint).name
        except Exception:  # noqa: BLE001
            name = ""
        if _looks_like_valid_filename(name):
            for d in image_search_dirs:
                p = d / name
                if _safe_exists(p):
                    return p

    # Try all image variants with numeric suffix.
    for d in image_search_dirs:
        for suffix in ["jpg", "jpeg", "png", "bmp", "gif"]:
            for idx in range(0, 10):
                p = d / f"{raw_id}_{idx}.{suffix}"
                if _safe_exists(p):
                    return p
    return None



def load_and_clean_records(
    dataset_root: Path,
    fake_json_rel: str,
    real_json_rel: str,
    image_dirs_rel: List[str],
    deduplicate_by_raw_id: bool,
    duplicate_policy: str,
    include_comments: bool,
    max_text_chars: int,
    snapshot_root: Path,
) -> tuple[List[NewsRecord], CleaningSummary, Path]:
    fake_path = dataset_root / fake_json_rel
    real_path = dataset_root / real_json_rel
    image_dirs = [dataset_root / p for p in image_dirs_rel]

    fake_rows = read_jsonl(fake_path)
    real_rows = read_jsonl(real_path)
    all_rows = [("fake", row) for row in fake_rows] + [("real", row) for row in real_rows]

    cleaned: List[NewsRecord] = []
    seen_idx: Dict[str, int] = {}
    removed_duplicates = 0
    removed_empty_text = 0

    uid_counters: Dict[str, int] = {}

    for src_split, row in all_rows:
        raw_id = str(row.get("id", "")).strip()
        if not raw_id:
            continue

        text = _normalize_text(str(row.get("content", "")))
        if include_comments:
            comments = _normalize_text(str(row.get("comments", "")))
            if comments:
                text = f"{text} [COMMENTS] {comments}".strip()
        if not text:
            removed_empty_text += 1
            continue
        text = text[:max_text_chars]

        label = int(row.get("label", 1 if src_split == "fake" else 0))
        domain_raw = str(row.get("category", "未知")).strip() or "未知"
        domain = CATEGORY_MAP.get(domain_raw, domain_raw)

        image_hint = _extract_first_image_path(row.get("piclists", None))
        resolved_image = _resolve_image(raw_id=raw_id, img_hint=image_hint, image_search_dirs=image_dirs)
        image_path = str(resolved_image) if resolved_image else ""
        has_image = int(resolved_image is not None)

        payload_for_hash = {
            "id": raw_id,
            "label": label,
            "domain": domain,
            "text": text,
            "image_path": image_path,
        }

        uid_key = f"{raw_id}#{src_split}"
        uid_idx = uid_counters.get(uid_key, 0)
        uid_counters[uid_key] = uid_idx + 1
        uid = uid_key if uid_idx == 0 else f"{uid_key}#{uid_idx}"

        record = NewsRecord(
            uid=uid,
            raw_id=raw_id,
            label=label,
            domain_raw=domain_raw,
            domain=domain,
            text=text,
            timestamp=str(row.get("timestamp", "")),
            comments=_normalize_text(str(row.get("comments", ""))),
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
    records_path = snapshot_dir / "clean_records.jsonl"
    summary_path = snapshot_dir / "cleaning_summary.json"

    write_jsonl(records_path, [asdict(r) for r in cleaned])
    write_json(summary_path, asdict(summary))

    return cleaned, summary, snapshot_dir
