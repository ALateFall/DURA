#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

import argparse
import csv
import json
import os
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a DURA checkpoint under missing-modality and image-noise robustness settings.")
    p.add_argument("--source-run-dir", type=str, required=True, help="Existing DURA run directory with run_config.json.")
    p.add_argument("--checkpoint-path", type=str, default="", help="Checkpoint path. Default: <source-run-dir>/checkpoints/dura_best.pt")
    p.add_argument("--model-name", type=str, default="", help="Display name stored in output tables.")
    p.add_argument("--name", type=str, default="", help="Output run dir name under paths.output_root.")
    p.add_argument("--dataset-root", type=str, default="", help="Optional dataset root override.")
    p.add_argument("--device", type=str, default="", help="Device override, e.g. cuda:0 or cpu.")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--threshold", type=float, default=-1.0, help="Use checkpoint threshold when < 0.")
    p.add_argument(
        "--nested-noise",
        action="store_true",
        help=(
            "Use nested noisy-image perturbations: +1/+2/+4 reuse prefixes of "
            "the same per-sample distractor sequence."
        ),
    )
    p.add_argument("--max-noise-images", type=int, default=4, help="Maximum noisy images for nested perturbation.")
    return p.parse_args()


def _select_device(device_str: str) -> torch.device:
    device_str = str(device_str)
    if device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        ordinal = int(device_str.split(":", 1)[1]) if ":" in device_str else 0
        visible = int(torch.cuda.device_count())
        if ordinal < 0 or ordinal >= visible:
            env = os.environ.get("CUDA_VISIBLE_DEVICES")
            raise ValueError(
                f"Requested device {device_str}, but only {visible} CUDA device(s) are visible "
                f"under CUDA_VISIBLE_DEVICES={env!r}."
            )
    return torch.device(device_str)


def _load_cfg_from_run(run_dir: Path) -> DURAConfig:
    payload = read_json(run_dir / "run_config.json")
    raw_cfg = payload.get("config", payload)
    return load_config_from_dict(raw_cfg)


def _metric_row(model_name: str, table: str, condition: str, threshold: float, metrics: Dict[str, object], stat: Dict[str, object]) -> Dict[str, object]:
    overall = metrics["overall"]
    per_class = metrics.get("per_class", {})
    real = per_class.get("real", {})
    fake = per_class.get("fake", {})
    return {
        "model": model_name,
        "table": table,
        "condition": condition,
        "threshold": float(threshold),
        "acc": float(overall.get("acc", 0.0)),
        "macro_f1": float(overall.get("macro_f1", 0.0)),
        "real_f1": float(real.get("f1", 0.0)),
        "fake_f1": float(fake.get("f1", 0.0)),
        "stat": stat,
        "metrics": metrics,
    }


def _write_rows_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "table", "condition", "threshold", "acc", "macro_f1", "real_f1", "fake_f1", "stat"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "table": row["table"],
                    "condition": row["condition"],
                    "threshold": f"{float(row['threshold']):.6f}",
                    "acc": f"{float(row['acc']):.6f}",
                    "macro_f1": f"{float(row['macro_f1']):.6f}",
                    "real_f1": f"{float(row['real_f1']):.6f}",
                    "fake_f1": f"{float(row['fake_f1']):.6f}",
                    "stat": json.dumps(row["stat"], ensure_ascii=False, sort_keys=True),
                }
            )


def _move_batch(batch: Dict[str, object], device: torch.device, non_blocking: bool) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=non_blocking)
        else:
            out[key] = value
    return out


def _collect_image_pool(loader, device: torch.device, non_blocking: bool) -> Tuple[torch.Tensor, List[int]]:
    feats: List[torch.Tensor] = []
    sample_indices: List[int] = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device=device, non_blocking=non_blocking)
            image_features = batch["image_features"].detach().cpu()
            image_mask = batch["image_mask"].detach().cpu()
            bsz, max_images = int(image_mask.size(0)), int(image_mask.size(1))
            for local_i in range(bsz):
                for slot_i in range(max_images):
                    if float(image_mask[local_i, slot_i].item()) > 0.5:
                        feats.append(image_features[local_i, slot_i].clone())
                        sample_indices.append(offset + local_i)
            offset += bsz
    if not feats:
        raise RuntimeError("No valid image features found in the test split; image-noise evaluation cannot run.")
    return torch.stack(feats, dim=0).contiguous(), sample_indices


def _apply_missing(
    batch: Dict[str, object],
    condition: str,
    rng: torch.Generator,
) -> Dict[str, int]:
    bsz = int(batch["label"].size(0))
    stat = {"n_drop_text": 0, "n_drop_image": 0}
    if condition == "clean":
        return stat

    if condition == "wo_image":
        drop_image = torch.ones(bsz, device=batch["label"].device, dtype=torch.bool)
        drop_text = torch.zeros_like(drop_image)
    elif condition == "wo_text":
        drop_text = torch.ones(bsz, device=batch["label"].device, dtype=torch.bool)
        drop_image = torch.zeros_like(drop_text)
    elif condition == "image_missing_50":
        drop_image = (torch.rand(bsz, generator=rng) < 0.5).to(batch["label"].device)
        drop_text = torch.zeros_like(drop_image)
    elif condition == "text_missing_50":
        drop_text = (torch.rand(bsz, generator=rng) < 0.5).to(batch["label"].device)
        drop_image = torch.zeros_like(drop_text)
    else:
        raise ValueError(f"Unsupported missing condition: {condition}")

    if bool(drop_image.any()):
        batch["image_features"] = batch["image_features"].clone()
        batch["image_mask"] = batch["image_mask"].clone()
        batch["image_features"][drop_image] = 0.0
        batch["image_mask"][drop_image] = 0.0
        stat["n_drop_image"] = int(drop_image.sum().item())

    if bool(drop_text.any()):
        batch["text_feature"] = batch["text_feature"].clone()
        batch["text_feature"][drop_text] = 0.0
        stat["n_drop_text"] = int(drop_text.sum().item())

    return stat


def _apply_noise_injection(
    batch: Dict[str, object],
    offset: int,
    n_noise: int,
    rng: random.Random,
    image_pool: torch.Tensor,
    pool_sample_indices: List[int],
    nested_max_noise: int = 0,
) -> Dict[str, int]:
    bsz = int(batch["label"].size(0))
    if n_noise <= 0:
        return {"n_injected": 0, "n_replaced_existing": 0}

    image_features = batch["image_features"].clone()
    image_mask = batch["image_mask"].clone()
    max_images = int(image_mask.size(1))
    n_pool = int(image_pool.size(0))
    n_injected = 0
    n_replaced = 0

    for local_i in range(bsz):
        global_i = offset + local_i
        empty_slots = [idx for idx in range(max_images) if float(image_mask[local_i, idx].item()) <= 0.5]
        valid_slots = [idx for idx in range(max_images) if float(image_mask[local_i, idx].item()) > 0.5]
        candidate_slots = empty_slots + list(reversed(valid_slots))
        if not candidate_slots:
            continue

        used_slots: set[int] = set()
        draw_count = max(int(n_noise), int(nested_max_noise))
        for noise_i in range(draw_count):
            if noise_i >= len(candidate_slots):
                break
            slot = candidate_slots[noise_i]
            donor = rng.randrange(n_pool)
            for _ in range(16):
                if pool_sample_indices[donor] != global_i:
                    break
                donor = rng.randrange(n_pool)
            if noise_i >= int(n_noise):
                continue
            if slot in used_slots:
                continue
            used_slots.add(slot)
            if float(image_mask[local_i, slot].item()) > 0.5:
                n_replaced += 1
            image_features[local_i, slot] = image_pool[donor].to(image_features.device)
            image_mask[local_i, slot] = 1.0
            n_injected += 1

    batch["image_features"] = image_features
    batch["image_mask"] = image_mask
    return {"n_injected": int(n_injected), "n_replaced_existing": int(n_replaced)}


def _evaluate_condition(
    model: DURAModel,
    loader,
    device: torch.device,
    threshold: float,
    non_blocking: bool,
    missing_condition: str = "clean",
    noise_count: int = 0,
    seed: int = 0,
    image_pool: torch.Tensor | None = None,
    pool_sample_indices: List[int] | None = None,
    nested_max_noise: int = 0,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    total_stat: Dict[str, int] = {
        "n_total": 0,
        "n_drop_text": 0,
        "n_drop_image": 0,
        "n_injected": 0,
        "n_replaced_existing": 0,
    }
    torch_rng = torch.Generator(device="cpu")
    torch_rng.manual_seed(int(seed))
    py_rng = random.Random(int(seed))
    offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device=device, non_blocking=non_blocking)
            if "text_feature" not in batch or "image_features" not in batch:
                raise RuntimeError("DURA robustness evaluation requires offline feature-cache batches.")

            bsz = int(batch["label"].size(0))
            total_stat["n_total"] += bsz

            miss_stat = _apply_missing(batch=batch, condition=missing_condition, rng=torch_rng)
            total_stat["n_drop_text"] += miss_stat["n_drop_text"]
            total_stat["n_drop_image"] += miss_stat["n_drop_image"]

            if int(noise_count) > 0:
                if image_pool is None or pool_sample_indices is None:
                    raise RuntimeError("image_pool and pool_sample_indices are required for noise injection.")
                noise_stat = _apply_noise_injection(
                    batch=batch,
                    offset=offset,
                    n_noise=int(noise_count),
                    rng=py_rng,
                    image_pool=image_pool,
                    pool_sample_indices=pool_sample_indices,
                    nested_max_noise=int(nested_max_noise),
                )
                total_stat["n_injected"] += noise_stat["n_injected"]
                total_stat["n_replaced_existing"] += noise_stat["n_replaced_existing"]

            out = model(batch)
            logits = out["logits"]
            prob_fake = torch.sigmoid(logits.reshape(-1)) if logits.dim() == 1 or logits.size(-1) == 1 else F.softmax(logits, dim=-1)[:, 1]
            y_true.extend(batch["label"].detach().cpu().tolist())
            y_prob.extend(prob_fake.detach().cpu().tolist())
            offset += bsz

    metrics = evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=threshold)
    stat = {
        **total_stat,
        "drop_text_ratio": float(total_stat["n_drop_text"] / max(1, total_stat["n_total"])),
        "drop_image_ratio": float(total_stat["n_drop_image"] / max(1, total_stat["n_total"])),
        "inject_ratio_per_sample": float(total_stat["n_injected"] / max(1, total_stat["n_total"])),
        "nested_max_noise": int(nested_max_noise),
    }
    return metrics, stat


def main():
    args = parse_args()

    global torch, F
    global ensure_dir, read_json, write_json
    global build_logger, set_seed, shutdown_loader, evaluate_predictions
    global load_config_from_dict, create_data_bundle, make_loader_for_uids
    global prepare_feature_cache, CLIPBackbone, DURAModel

    import torch
    import torch.nn.functional as F

    from dura.common.io import ensure_dir, read_json, write_json
    from dura.common.logger import build_logger
    from dura.common.seed import set_seed
    from dura.config import load_config_from_dict
    from dura.data import create_data_bundle, make_loader_for_uids
    from dura.data_utils.loader import shutdown_loader
    from dura.eval_utils.metrics import evaluate_predictions
    from dura.feature_cache import prepare_feature_cache
    from dura.model import CLIPBackbone, DURAModel

    source_run_dir = Path(args.source_run_dir)
    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source run dir not found: {source_run_dir}")

    cfg = _load_cfg_from_run(source_run_dir)
    if args.dataset_root:
        cfg.paths.dataset_root = str(args.dataset_root)
    if args.device:
        cfg.experiment.device = str(args.device)
    device = _select_device(cfg.experiment.device)
    set_seed(int(args.seed))

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else source_run_dir / "checkpoints" / "dura_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model_name = args.model_name or cfg.experiment.ablation or source_run_dir.name
    out_name = args.name or f"dura_robustness_{source_run_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = ensure_dir(Path(cfg.paths.output_root) / out_name)
    logger = build_logger(out_dir / "logs" / "run.log")

    write_json(
        out_dir / "run_config.json",
        {
            "config": asdict(cfg),
            "extra": {
                "source_run_dir": str(source_run_dir),
                "checkpoint_path": str(checkpoint_path),
                "model_name": model_name,
                "seed": int(args.seed),
                "device": str(device),
                "nested_noise": bool(args.nested_noise),
                "max_noise_images": int(args.max_noise_images),
            },
        },
    )

    logger.info("Source run dir: %s", source_run_dir)
    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Model name: %s", model_name)
    logger.info("Output dir: %s", out_dir)
    logger.info("Device: %s", device)
    logger.info("Nested noise: %s | max_noise_images=%s", bool(args.nested_noise), int(args.max_noise_images))

    backbone = CLIPBackbone(cfg=cfg.clip, target_dim=cfg.model.proj_dim, logger=logger)
    data_bundle = create_data_bundle(cfg=cfg, run_dir=out_dir, logger=logger, feature_cache=None, build_loaders=False)
    all_uids = sorted(set(data_bundle.split.train_ids + data_bundle.split.val_ids + data_bundle.split.test_ids))
    feature_cache = prepare_feature_cache(
        cfg=cfg,
        records_by_uid=data_bundle.records_by_uid,
        all_uids=all_uids,
        backbone=backbone,
        device=device,
        logger=logger,
    )
    if feature_cache is None:
        raise RuntimeError("Feature cache is required for DURA robustness evaluation.")
    test_loader = make_loader_for_uids(cfg, data_bundle.records_by_uid, data_bundle.split.test_ids, is_train=False, feature_cache=feature_cache)

    domain_loss_enabled = float(cfg.loss.private_domain_weight) > 0.0
    num_domains = len(data_bundle.source_domains) if domain_loss_enabled else 0
    model = DURAModel(
        backbone=backbone,
        lora_cfg=cfg.lora,
        model_cfg=cfg.model,
        ablation=cfg.experiment.ablation,
        num_domains=num_domains,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()

    threshold = float(checkpoint.get("threshold", 0.5)) if float(args.threshold) < 0 else float(args.threshold)
    checkpoint_epoch = int(checkpoint.get("epoch", 0))
    logger.info("Threshold: %.6f | checkpoint_epoch=%s", threshold, checkpoint_epoch)

    image_pool, pool_sample_indices = _collect_image_pool(
        loader=test_loader,
        device=device,
        non_blocking=bool(cfg.data.non_blocking_transfer),
    )
    logger.info("Image pool: %s valid image slots", int(image_pool.size(0)))

    rows: List[Dict[str, object]] = []
    missing_conditions = [
        ("Clean", "clean"),
        ("w/o Image", "wo_image"),
        ("w/o Text", "wo_text"),
        ("Image Missing 50%", "image_missing_50"),
        ("Text Missing 50%", "text_missing_50"),
    ]
    for idx, (display, condition) in enumerate(missing_conditions):
        metrics, stat = _evaluate_condition(
            model=model,
            loader=test_loader,
            device=device,
            threshold=threshold,
            non_blocking=bool(cfg.data.non_blocking_transfer),
            missing_condition=condition,
            seed=int(args.seed) + idx * 101,
        )
        row = _metric_row(model_name, "missing_modality", display, threshold, metrics, stat)
        rows.append(row)
        logger.info("[Missing] %s acc=%.6f macro_f1=%.6f", display, row["acc"], row["macro_f1"])

    noise_conditions = [
        ("Clean", 0),
        ("+1 Noise Image", 1),
        ("+2 Noise Images", 2),
        ("+4 Noise Images", 4),
    ]
    nested_max_noise = int(args.max_noise_images) if bool(args.nested_noise) else 0
    noise_base_seed = int(args.seed) + 1009
    for idx, (display, noise_count) in enumerate(noise_conditions):
        metrics, stat = _evaluate_condition(
            model=model,
            loader=test_loader,
            device=device,
            threshold=threshold,
            non_blocking=bool(cfg.data.non_blocking_transfer),
            missing_condition="clean",
            noise_count=int(noise_count),
            seed=noise_base_seed if bool(args.nested_noise) else int(args.seed) + 1009 + idx * 103,
            image_pool=image_pool,
            pool_sample_indices=pool_sample_indices,
            nested_max_noise=nested_max_noise,
        )
        row = _metric_row(model_name, "image_noise", display, threshold, metrics, stat)
        rows.append(row)
        logger.info("[Noise] %s acc=%.6f macro_f1=%.6f", display, row["acc"], row["macro_f1"])

    _write_rows_csv(out_dir / "robustness_rows.csv", rows)
    payload = {
        "run_dir": str(out_dir),
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": str(checkpoint_path),
        "model_name": model_name,
        "ablation": cfg.experiment.ablation,
        "checkpoint_epoch": checkpoint_epoch,
        "threshold": threshold,
        "seed": int(args.seed),
        "nested_noise": bool(args.nested_noise),
        "max_noise_images": int(args.max_noise_images),
        "rows": rows,
        "detail_files": {
            "csv": str(out_dir / "robustness_rows.csv"),
        },
    }
    write_json(out_dir / "summary.json", payload)
    shutdown_loader(test_loader)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
