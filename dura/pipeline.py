from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch

from dura.pipeline_utils.dare_merge import merge_source_adapters_with_dare
from dura.common.io import ensure_dir, write_json
from dura.common.logger import build_logger
from dura.common.seed import set_seed
from dura.common.tb import create_summary_writer
from dura.data_utils.loader import shutdown_loader

from dura.config import DURAConfig, private_domain_loss_enabled
from dura.data import create_data_bundle, domain_specific_loaders, make_loader_for_uids
from dura.feature_cache import prepare_feature_cache
from dura.model import CLIPBackbone, DURAModel, DomainLoRAModel
from dura.train import build_class_weights, train_source_domain_lora, train_stage3


def _select_device(device_str: str) -> torch.device:
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(str(device_str))


def _build_run_name(cfg: DURAConfig) -> str:
    if cfg.experiment.name:
        return cfg.experiment.name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dura_{cfg.experiment.mode}_{stamp}"


def run_pipeline(cfg: DURAConfig) -> Dict[str, object]:
    run_dir = ensure_dir(Path(cfg.paths.output_root) / _build_run_name(cfg))
    logger = build_logger(run_dir / "logs" / "run.log")
    set_seed(int(cfg.experiment.seed))
    device = _select_device(cfg.experiment.device)
    logger.info("Run dir: %s", run_dir)
    logger.info("Device: %s", device)
    logger.info("Seed: %s", int(cfg.experiment.seed))

    if cfg.experiment.enable_tensorboard:
        tb_writer, tb_err = create_summary_writer(run_dir / "tensorboard")
        if tb_err:
            logger.info("TensorBoard disabled: %s", tb_err)
            tb_writer = None
        else:
            logger.info("TensorBoard dir: %s", run_dir / "tensorboard")
    else:
        tb_writer = None
        logger.info("TensorBoard disabled by config")

    write_json(run_dir / "run_config.json", {"config": asdict(cfg), "extra": {"device": str(device)}})

    backbone = CLIPBackbone(cfg=cfg.clip, target_dim=cfg.model.proj_dim, logger=logger)
    backbone_info = backbone.info()

    logger.info("Preparing DURA data bundle")
    data_bundle = create_data_bundle(cfg=cfg, run_dir=run_dir, logger=logger, feature_cache=None, build_loaders=False)
    feature_cache_uids = sorted(set(data_bundle.split.train_ids + data_bundle.split.val_ids + data_bundle.split.test_ids))
    logger.info("Preparing DURA feature cache")
    feature_cache = prepare_feature_cache(
        cfg=cfg,
        records_by_uid=data_bundle.records_by_uid,
        all_uids=feature_cache_uids,
        backbone=backbone,
        device=device,
        logger=logger,
    )
    if feature_cache is not None:
        logger.info("DURA feature cache enabled | path=%s", feature_cache.get("cache_path", ""))
    data_bundle.train_loader = make_loader_for_uids(cfg, data_bundle.records_by_uid, data_bundle.split.train_ids, is_train=True, feature_cache=feature_cache)
    data_bundle.val_loader = make_loader_for_uids(cfg, data_bundle.records_by_uid, data_bundle.split.val_ids, is_train=False, feature_cache=feature_cache)
    data_bundle.test_loader = make_loader_for_uids(cfg, data_bundle.records_by_uid, data_bundle.split.test_ids, is_train=False, feature_cache=feature_cache)

    class_weights = build_class_weights(data_bundle.records_by_uid, data_bundle.split.train_ids, cfg)
    if class_weights is not None:
        logger.info("DURA class weights (real,fake) = %s", class_weights.tolist())
    domain_loss_enabled = private_domain_loss_enabled(cfg)
    domain_to_idx = {domain: idx for idx, domain in enumerate(sorted(data_bundle.source_domains))} if domain_loss_enabled else {}

    source_adapter_states: Dict[str, Dict[str, torch.Tensor]] = {}
    domain_sizes: Dict[str, int] = {}
    for domain in data_bundle.source_domains:
        train_loader, val_loader, d_train_ids = domain_specific_loaders(
            cfg=cfg,
            records_by_uid=data_bundle.records_by_uid,
            train_ids=data_bundle.split.train_ids,
            val_ids=data_bundle.split.val_ids,
            domain=domain,
            feature_cache=feature_cache,
        )
        if train_loader is None:
            logger.info("[DURA-Stage1] Skip domain=%s because train split has no samples", domain)
            continue
        domain_sizes[domain] = len(d_train_ids)
        logger.info("[DURA-Stage1] Train source LoRA for domain=%s | n_train=%s", domain, len(d_train_ids))
        stage1_model = DomainLoRAModel(backbone=backbone, lora_cfg=cfg.lora, model_cfg=cfg.model)
        adapter_state = train_source_domain_lora(
            cfg=cfg,
            model=stage1_model,
            train_loader=train_loader,
            val_loader=val_loader,
            domain=domain,
            device=device,
            out_dir=run_dir / "checkpoints",
            logger=logger,
            class_weights=class_weights,
            tb_writer=tb_writer,
        )
        source_adapter_states[domain] = adapter_state
        shutdown_loader(train_loader)
        shutdown_loader(val_loader)

    shared_adapter_state = merge_source_adapters_with_dare(
        cfg=cfg,
        source_adapter_states=source_adapter_states,
        out_dir=run_dir / "checkpoints",
        logger=logger,
        domain_sizes=domain_sizes,
    )

    dura_model = DURAModel(
        backbone=backbone,
        lora_cfg=cfg.lora,
        model_cfg=cfg.model,
        ablation=cfg.experiment.ablation,
        num_domains=len(domain_to_idx) if domain_loss_enabled else 0,
    )
    dura_model.load_shared_adapter_state(shared_adapter_state)
    dura_model.freeze_shared_stream()
    stage3_result = train_stage3(
        cfg=cfg,
        model=dura_model,
        train_loader=data_bundle.train_loader,
        val_loader=data_bundle.val_loader,
        test_loader=data_bundle.test_loader,
        device=device,
        out_dir=run_dir / "checkpoints",
        logger=logger,
        class_weights=class_weights,
        tb_writer=tb_writer,
        domain_to_idx=(domain_to_idx if domain_loss_enabled else None),
    )

    summary = {
        "run_dir": str(run_dir),
        "mode": cfg.experiment.mode,
        "ablation": cfg.experiment.ablation,
        "backbone": {
            "provider": backbone_info.provider,
            "model_name": backbone_info.model_name,
            "output_dim": backbone_info.output_dim,
            "load_errors": backbone_info.load_errors,
        },
        "feature_cache": {
            "enabled": feature_cache is not None,
            "path": (feature_cache.get("cache_path") if feature_cache is not None else None),
        },
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "source_domains": list(data_bundle.source_domains),
        "best_epoch": stage3_result.best_epoch,
        "best_threshold": stage3_result.best_threshold,
        "best_val_score": stage3_result.best_val_score,
        "val": {"threshold": stage3_result.best_threshold, **stage3_result.val_metrics},
        "test": {"threshold": stage3_result.best_threshold, **stage3_result.test_metrics},
        "detail_files": {
            "test_per_domain_csv": str(run_dir / "checkpoints" / "test_per_domain.csv"),
            "predictions_val_json": str(stage3_result.val_predictions_path),
            "predictions_test_json": str(stage3_result.test_predictions_path),
        },
        "checkpoint": str(stage3_result.checkpoint_path),
    }
    write_json(run_dir / "summary.json", summary)
    logger.info("DURA summary saved | path=%s", run_dir / "summary.json")

    if tb_writer is not None:
        try:
            tb_writer.flush()
            tb_writer.close()
        except Exception:
            pass

    shutdown_loader(data_bundle.train_loader)
    shutdown_loader(data_bundle.val_loader)
    shutdown_loader(data_bundle.test_loader)
    return summary
