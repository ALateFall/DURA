from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, *args, **kwargs):
        return x

from dura.common.hashing import sha256_obj
from dura.common.io import ensure_dir, read_json, write_json

from dura.config import DURAConfig
from dura.data import DuraNewsRecord, build_loader
from dura.model import CLIPBackbone


_COMPAT_KEYS = (
    "dataset_name",
    "dataset_root",
    "provider",
    "model_name",
    "output_dim",
    "max_text_length",
    "image_size",
    "max_images_per_post",
)


def _records_fingerprint(records_by_uid: Dict[str, DuraNewsRecord], uids: List[str]) -> str:
    payload = {
        "uids": list(sorted(uids)),
        "rows": {uid: records_by_uid[uid].row_hash for uid in uids},
    }
    return sha256_obj(payload)



def _cache_identity(
    cfg: DURAConfig,
    backbone: CLIPBackbone,
    records_by_uid: Dict[str, DuraNewsRecord],
    uids: List[str],
) -> tuple[str, Dict[str, object]]:
    info = backbone.info()
    payload = {
        "dataset_name": str(cfg.data.dataset_name),
        "dataset_root": str(Path(cfg.paths.dataset_root).resolve()),
        "provider": info.provider,
        "model_name": info.model_name,
        "output_dim": info.output_dim,
        "max_text_length": cfg.clip.max_text_length,
        "image_size": cfg.clip.image_size,
        "max_images_per_post": cfg.data.max_images_per_post,
        "dataset_fingerprint": _records_fingerprint(records_by_uid, uids),
    }
    key = sha256_obj(payload)[:24]
    return key, payload



def _cache_root(cfg: DURAConfig) -> Path:
    if cfg.paths.cache_dir:
        return ensure_dir(Path(cfg.paths.cache_dir))
    return ensure_dir(Path(cfg.paths.output_root) / "feature_cache_dura")



def _load_feature_cache(path: Path, logger):
    obj = torch.load(path, map_location="cpu")
    required = {"uids", "text_features", "image_features", "image_mask"}
    if not isinstance(obj, dict) or not required.issubset(set(obj.keys())):
        raise RuntimeError(f"Invalid DURA feature cache format: {path}")
    uids = list(obj["uids"])
    text_features = obj["text_features"].float().cpu().contiguous()
    image_features = obj["image_features"].float().cpu().contiguous()
    image_mask = obj["image_mask"].float().cpu().contiguous()
    n = len(uids)
    if text_features.size(0) != n or image_features.size(0) != n or image_mask.size(0) != n:
        raise RuntimeError(f"DURA feature cache shape mismatch: {path}")
    uid_to_index = {uid: i for i, uid in enumerate(uids)}
    logger.info(
        "Loaded DURA feature cache | path=%s n=%s text_dim=%s max_images=%s",
        path,
        n,
        int(text_features.size(-1)),
        int(image_features.size(1)),
    )
    return {
        "cache_path": str(path),
        "uids": uids,
        "uid_to_index": uid_to_index,
        "text_features": text_features,
        "image_features": image_features,
        "image_mask": image_mask,
    }



def _slice_feature_cache(cache: dict, uids: List[str]) -> dict:
    idx = torch.tensor([cache["uid_to_index"][uid] for uid in uids], dtype=torch.long)
    sliced_uids = list(uids)
    return {
        "cache_path": cache.get("cache_path", ""),
        "uids": sliced_uids,
        "uid_to_index": {uid: i for i, uid in enumerate(sliced_uids)},
        "text_features": cache["text_features"].index_select(0, idx).contiguous(),
        "image_features": cache["image_features"].index_select(0, idx).contiguous(),
        "image_mask": cache["image_mask"].index_select(0, idx).contiguous(),
    }



def _build_feature_cache(
    cfg: DURAConfig,
    records_by_uid: Dict[str, DuraNewsRecord],
    uids: List[str],
    backbone: CLIPBackbone,
    device: torch.device,
    cache_file: Path,
    logger,
):
    logger.info("Building DURA feature cache | n=%s batch_size=%s", len(uids), cfg.data.feature_cache_batch_size)
    num_workers = int(getattr(cfg, "_effective_num_workers", cfg.data.num_workers))
    persistent_workers = bool(getattr(cfg, "_effective_persistent_workers", cfg.data.persistent_workers and num_workers > 0))
    loader = build_loader(
        records_by_uid=records_by_uid,
        uids=uids,
        batch_size=cfg.data.feature_cache_batch_size,
        image_size=cfg.clip.image_size,
        max_images_per_post=cfg.data.max_images_per_post,
        sampler_strategy="none",
        num_workers=num_workers,
        pin_memory=cfg.data.pin_memory,
        shuffle=False,
        persistent_workers=persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        feature_cache=None,
    )
    backbone.to(device)
    backbone.eval()

    uid_buf: List[str] = []
    text_buf: List[torch.Tensor] = []
    image_buf: List[torch.Tensor] = []
    mask_buf: List[torch.Tensor] = []

    iterator = tqdm(loader, total=len(loader), desc="DURA-Cache", leave=False)
    with torch.inference_mode():
        for batch in iterator:
            uid_buf.extend(list(batch["uid"]))
            images = batch["images"].to(device)
            image_mask = batch["image_mask"].to(device)
            text_feat, image_feat = backbone.encode_batch(batch={"text": batch["text"], "images": images, "image_mask": image_mask})
            text_buf.append(text_feat.detach().cpu())
            image_buf.append(image_feat.detach().cpu())
            mask_buf.append(image_mask.detach().cpu())

    payload = {
        "uids": uid_buf,
        "text_features": torch.cat(text_buf, dim=0).contiguous(),
        "image_features": torch.cat(image_buf, dim=0).contiguous(),
        "image_mask": torch.cat(mask_buf, dim=0).contiguous(),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".pt.tmp")
    if tmp_file.exists():
        tmp_file.unlink()
    torch.save(payload, tmp_file)
    tmp_file.replace(cache_file)
    logger.info("Saved DURA feature cache | path=%s", cache_file)
    return _load_feature_cache(cache_file, logger)



def _compat_signature(meta: Dict[str, object]) -> Dict[str, object]:
    return {k: meta.get(k) for k in _COMPAT_KEYS}


def _is_compat_meta(meta: Dict[str, object], identity: Dict[str, object]) -> bool:
    for key in _COMPAT_KEYS:
        meta_val = meta.get(key, None)
        want_val = identity.get(key, None)
        # Backward-compatible reuse for legacy caches that were saved
        # before dataset_name/dataset_root were recorded.
        if key in {"dataset_name", "dataset_root"} and meta_val in {None, ""}:
            continue
        if meta_val != want_val:
            return False
    return True



def _find_superset_cache(
    cfg: DURAConfig,
    requested_uids: List[str],
    identity: Dict[str, object],
    logger,
) -> Optional[dict]:
    cache_root = _cache_root(cfg)
    requested_set = set(requested_uids)
    for meta_file in sorted(cache_root.glob("*/meta.json")):
        cache_file = meta_file.with_name("features.pt")
        if not cache_file.exists():
            continue
        try:
            meta = read_json(meta_file)
        except Exception as exc:  # noqa: BLE001
            logger.info("Skip DURA cache meta due to read error | path=%s err=%s", meta_file, exc)
            continue
        if not isinstance(meta, dict) or (not _is_compat_meta(meta, identity)):
            continue
        try:
            cache = _load_feature_cache(cache_file, logger)
        except Exception as exc:  # noqa: BLE001
            logger.info("Skip DURA cache due to load error | path=%s err=%s", cache_file, exc)
            continue
        if requested_set.issubset(set(cache["uids"])):
            logger.info(
                "Reusing compatible DURA superset cache | path=%s requested=%s cached=%s",
                cache_file,
                len(requested_uids),
                len(cache["uids"]),
            )
            return _slice_feature_cache(cache, requested_uids)
    return None



def prepare_feature_cache(
    cfg: DURAConfig,
    records_by_uid: Dict[str, DuraNewsRecord],
    all_uids: List[str],
    backbone: CLIPBackbone | None,
    device: torch.device | None,
    logger,
):
    if not cfg.data.use_feature_cache:
        return None
    if not cfg.clip.freeze_backbone:
        logger.info("DURA feature cache disabled because clip.freeze_backbone=false")
        return None
    if backbone is None or device is None:
        logger.info("DURA feature cache skipped because backbone/device unavailable")
        return None

    cache_key, identity = _cache_identity(cfg=cfg, backbone=backbone, records_by_uid=records_by_uid, uids=all_uids)
    cache_dir = ensure_dir(_cache_root(cfg) / cache_key)
    cache_file = cache_dir / "features.pt"
    meta_file = cache_dir / "meta.json"

    existing_meta = read_json(meta_file) if meta_file.exists() else {}
    if cache_file.exists() and meta_file.exists() and existing_meta == identity and (not cfg.data.force_rebuild_feature_cache):
        return _load_feature_cache(cache_file, logger)

    if not cfg.data.force_rebuild_feature_cache:
        superset_cache = _find_superset_cache(cfg=cfg, requested_uids=all_uids, identity=identity, logger=logger)
        if superset_cache is not None:
            return superset_cache

    if not cfg.data.build_feature_cache_if_missing:
        logger.info("DURA feature cache missing and build_feature_cache_if_missing=false, fallback to online features")
        return None

    write_json(meta_file, identity)
    return _build_feature_cache(
        cfg=cfg,
        records_by_uid=records_by_uid,
        uids=all_uids,
        backbone=backbone,
        device=device,
        cache_file=cache_file,
        logger=logger,
    )
