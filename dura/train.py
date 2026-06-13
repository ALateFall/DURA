from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    from tqdm.auto import tqdm
except Exception:  # noqa: BLE001
    def tqdm(x, *args, **kwargs):
        return x

from dura.common.checkpoint import save_checkpoint
from dura.common.io import write_json
from dura.config import DURAConfig, private_domain_loss_enabled
from dura.eval_utils.metrics import evaluate_predictions, per_domain_metrics
from dura.eval_utils.reports import write_metrics_report, write_per_domain_csv
from dura.eval_utils.threshold import ThresholdSearchResult, find_best_threshold, score_from_metrics
from dura.model import DURAModel, DomainLoRAModel
from dura.modules.losses import classification_loss, orthogonality_loss


@dataclass
class EvalOutputs:
    y_true: List[int]
    y_prob: List[float]
    domains: List[str]
    uids: List[str]
    logits: List[List[float]]


@dataclass
class Stage3Result:
    best_epoch: int
    best_threshold: float
    best_val_score: float
    val_metrics: Dict[str, object]
    test_metrics: Dict[str, object]
    per_domain: Dict[str, Dict[str, object]]
    checkpoint_path: Path
    val_predictions_path: Path
    test_predictions_path: Path


def _loader_item_count(loader) -> int:
    if loader is None:
        return 0
    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        try:
            return int(len(dataset))
        except Exception:  # noqa: BLE001
            pass
    return 0


class _TrainableEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=(1.0 - self.decay))

    @contextmanager
    def apply_to(self, model: torch.nn.Module):
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name not in self.shadow:
                    continue
                backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])


def _clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _orth_weight_for_epoch(cfg: DURAConfig, epoch_idx: int) -> float:
    if cfg.experiment.ablation in {"w/o_orth", "w/o_shared", "w/o_private"}:
        return 0.0
    base = float(cfg.loss.orth_weight)
    warmup_epochs = int(cfg.train.orth_warmup_epochs)
    init_factor = float(cfg.train.orth_warmup_init_factor)
    if warmup_epochs <= 0:
        return base
    progress = min(1.0, float(epoch_idx + 1) / float(max(1, warmup_epochs)))
    return base * (init_factor + (1.0 - init_factor) * progress)


def _make_scheduler(optimizer, total_steps: int, warmup_ratio: float, min_lr_scale: float = 0.0):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda)


def _compute_class_weights(records_by_uid, train_ids, strategy: str) -> torch.Tensor | None:
    if strategy == "none":
        return None
    n0 = sum(1 for uid in train_ids if records_by_uid[uid].label == 0)
    n1 = sum(1 for uid in train_ids if records_by_uid[uid].label == 1)
    n0 = max(1, n0)
    n1 = max(1, n1)
    if strategy == "inverse":
        w0 = 1.0 / n0
        w1 = 1.0 / n1
    elif strategy == "sqrt_inverse":
        w0 = 1.0 / (n0 ** 0.5)
        w1 = 1.0 / (n1 ** 0.5)
    else:
        raise ValueError(f"Unsupported class_weighting: {strategy}")
    s = w0 + w1
    return torch.tensor([w0 / s, w1 / s], dtype=torch.float32)


def _compute_cls_loss(logits, labels, cfg: DURAConfig, class_weights):
    return classification_loss(
        logits=logits,
        labels=labels,
        kind=cfg.loss.classification,
        label_smoothing=cfg.loss.label_smoothing,
        focal_gamma=cfg.loss.focal_gamma,
        reduction="mean",
        class_weights=class_weights,
        focal_alpha_pos=cfg.loss.focal_alpha_pos,
    )


def _compute_reliability_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    visual_reliability: torch.Tensor | None,
    has_visual: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    if visual_reliability is None or has_visual is None:
        return logits.new_zeros(())
    mask = has_visual.float()
    if torch.count_nonzero(mask).item() == 0:
        return logits.new_zeros(())
    binary_logits = logits.reshape(-1) if logits.dim() == 1 or logits.size(-1) == 1 else logits[:, 1] - logits[:, 0]
    base_bce = F.binary_cross_entropy_with_logits(binary_logits, labels.float(), reduction="none")
    lam = torch.clamp(visual_reliability.float(), min=0.0, max=1e4)
    loss_vec = 0.5 * lam * base_bce - 0.5 * mask * torch.log(lam + float(eps))
    return (loss_vec * mask).mean()


def _compute_domain_loss(logits: torch.Tensor | None, domain_targets: torch.Tensor | None) -> torch.Tensor:
    if logits is None or domain_targets is None:
        if torch.is_tensor(logits):
            return logits.new_zeros(())
        return torch.tensor(0.0)
    return F.cross_entropy(logits, domain_targets)


def _move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def train_source_domain_lora(
    cfg: DURAConfig,
    model: DomainLoRAModel,
    train_loader,
    val_loader,
    domain: str,
    device: torch.device,
    out_dir: Path,
    logger,
    class_weights: Optional[torch.Tensor] = None,
    tb_writer=None,
) -> Dict[str, torch.Tensor]:
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.train.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * cfg.train.stage1_epochs)
    scheduler = _make_scheduler(optimizer, total_steps, cfg.optim.warmup_ratio, cfg.optim.min_lr / max(cfg.optim.lr, 1e-12))
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg.train.stage1_epochs):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        train_iter = tqdm(train_loader, total=steps_per_epoch, disable=(not cfg.train.show_progress), desc=f"DURA-S1-{domain} E{epoch+1}/{cfg.train.stage1_epochs}", leave=False)
        for step, batch in enumerate(train_iter):
            if cfg.train.max_steps_per_epoch > 0 and step >= cfg.train.max_steps_per_epoch:
                break
            batch = _move_batch(batch, device)
            out = model(batch)
            labels = batch["label"]
            loss = _compute_cls_loss(out["logits"], labels, cfg, class_weights)
            optimizer.zero_grad()
            loss.backward()
            if cfg.optim.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.optim.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            bs = int(labels.size(0))
            train_loss_sum += float(loss.detach().item()) * bs
            seen += bs

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = _move_batch(batch, device)
                out = model(batch)
                labels = batch["label"]
                loss = _compute_cls_loss(out["logits"], labels, cfg, class_weights)
                bs = int(labels.size(0))
                val_loss_sum += float(loss.detach().item()) * bs
                val_seen += bs
        val_loss = val_loss_sum / max(1, val_seen)
        train_loss = train_loss_sum / max(1, seen)
        logger.info("[DURA-Stage1][%s] epoch=%s train_loss=%.6f val_loss=%.6f", domain, epoch + 1, train_loss, val_loss)
        if tb_writer is not None:
            tb_writer.add_scalar(f"dura_stage1/{domain}/train_loss", train_loss, epoch + 1)
            tb_writer.add_scalar(f"dura_stage1/{domain}/val_loss", val_loss, epoch + 1)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.adapter_state()

    if best_state is None:
        best_state = model.adapter_state()
    save_checkpoint(out_dir / f"stage1_{domain}_adapter.pt", {"domain": domain, "adapter_state": best_state, "best_val_loss": best_val_loss})
    return best_state


def evaluate_loader(model: DURAModel, loader, device: torch.device) -> EvalOutputs:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    domains: List[str] = []
    uids: List[str] = []
    logits_list: List[List[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            out = model(batch)
            logits = out["logits"]
            prob = torch.sigmoid(logits.reshape(-1)) if logits.dim() == 1 or logits.size(-1) == 1 else torch.softmax(logits, dim=-1)[:, 1]
            y_true.extend(batch["label"].detach().cpu().numpy().tolist())
            y_prob.extend(prob.detach().cpu().numpy().tolist())
            domains.extend(list(batch["domain"]))
            uids.extend(list(batch["uid"]))
            logits_list.extend(out["logits"].detach().cpu().numpy().tolist())
    return EvalOutputs(y_true=y_true, y_prob=y_prob, domains=domains, uids=uids, logits=logits_list)


def _write_prediction_json(path: Path, outputs: EvalOutputs, threshold: float, best_epoch: int) -> None:
    write_json(
        path,
        {
            "threshold": float(threshold),
            "best_epoch": int(best_epoch),
            "rows": [
                {
                    "uid": uid,
                    "domain": domain,
                    "y_true": int(y),
                    "y_prob": float(p),
                    "logits": logit,
                }
                for uid, domain, y, p, logit in zip(
                    outputs.uids,
                    outputs.domains,
                    outputs.y_true,
                    outputs.y_prob,
                    outputs.logits,
                )
            ],
        },
    )


def train_stage3(
    cfg: DURAConfig,
    model: DURAModel,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    out_dir: Path,
    logger,
    class_weights: Optional[torch.Tensor] = None,
    tb_writer=None,
    domain_to_idx: Dict[str, int] | None = None,
) -> Stage3Result:
    if _loader_item_count(val_loader) == 0:
        raise RuntimeError("Validation split is empty. Provide a non-empty validation split for threshold selection and early stopping.")
    if _loader_item_count(test_loader) == 0:
        raise RuntimeError("Test split is empty. Provide a non-empty test split for final evaluation.")

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    steps_per_epoch = len(train_loader)
    if cfg.train.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, cfg.train.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * cfg.train.stage3_epochs)
    scheduler = _make_scheduler(optimizer, total_steps, cfg.optim.warmup_ratio, cfg.optim.min_lr / max(cfg.optim.lr, 1e-12))

    best_monitor = -1e9 if cfg.train.monitor_mode == "max" else 1e9
    best_state = None
    best_threshold = 0.5
    best_val_metrics: Dict[str, object] = {}
    bad_epochs = 0
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp and device.type == "cuda"))
    ema = _TrainableEMA(model, cfg.train.ema_decay) if cfg.train.ema_enabled else None
    reliability_enabled = bool(float(cfg.loss.reliability_weight) > 0.0)
    domain_private_enabled = bool(private_domain_loss_enabled(cfg) and domain_to_idx)

    for epoch in range(cfg.train.stage3_epochs):
        model.train()
        current_orth_weight = _orth_weight_for_epoch(cfg, epoch)
        train_loss_sum = 0.0
        train_cls_sum = 0.0
        train_orth_sum = 0.0
        train_rel_sum = 0.0
        train_domain_private_sum = 0.0
        seen = 0
        train_iter = tqdm(train_loader, total=steps_per_epoch, disable=(not cfg.train.show_progress), desc=f"DURA-S3 E{epoch+1}/{cfg.train.stage3_epochs}", leave=False)
        for step, batch in enumerate(train_iter):
            if cfg.train.max_steps_per_epoch > 0 and step >= cfg.train.max_steps_per_epoch:
                break
            batch = _move_batch(batch, device)
            labels = batch["label"]
            domain_targets = (
                torch.tensor([domain_to_idx[d] for d in batch["domain"]], device=device, dtype=torch.long)
                if domain_private_enabled
                else None
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=bool(cfg.train.amp and device.type == "cuda")):
                out = model(batch)
                cls_loss = _compute_cls_loss(out["logits"], labels, cfg, class_weights)
                orth_loss = orthogonality_loss(out["z_shared"], out["z_private"])
                rel_loss = (
                    _compute_reliability_loss(
                        logits=out["logits"],
                        labels=labels,
                        visual_reliability=out.get("global_visual_reliability"),
                        has_visual=out.get("has_visual"),
                        eps=float(cfg.loss.reliability_eps),
                    )
                    if reliability_enabled
                    else cls_loss.new_zeros(())
                )
                private_domain_loss = (
                    _compute_domain_loss(out.get("private_domain_logits"), domain_targets).to(cls_loss.device)
                    if domain_private_enabled
                    else cls_loss.new_zeros(())
                )
                loss = (
                    cls_loss
                    + current_orth_weight * orth_loss
                    + (float(cfg.loss.reliability_weight) * rel_loss if reliability_enabled else 0.0)
                    + (float(cfg.loss.private_domain_weight) * private_domain_loss if domain_private_enabled else 0.0)
                )
            scaler.scale(loss).backward()
            if cfg.optim.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.optim.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            bs = int(labels.size(0))
            train_loss_sum += float(loss.detach().item()) * bs
            train_cls_sum += float(cls_loss.detach().item()) * bs
            train_orth_sum += float(orth_loss.detach().item()) * bs
            train_rel_sum += float(rel_loss.detach().item()) * bs
            train_domain_private_sum += float(private_domain_loss.detach().item()) * bs
            seen += bs

        train_loss = train_loss_sum / max(1, seen)
        train_cls = train_cls_sum / max(1, seen)
        train_orth = train_orth_sum / max(1, seen)
        train_rel = train_rel_sum / max(1, seen)
        train_domain_private = train_domain_private_sum / max(1, seen)

        if ema is not None:
            with ema.apply_to(model):
                val_out = evaluate_loader(model, val_loader, device)
                improved_state = _clone_state_dict(model)
        else:
            val_out = evaluate_loader(model, val_loader, device)
            improved_state = _clone_state_dict(model)

        threshold_res = find_best_threshold(
            val_out.y_true,
            val_out.y_prob,
            cfg.eval.threshold_metric,
            cfg.eval.threshold_min,
            cfg.eval.threshold_max,
            cfg.eval.threshold_step,
        )
        val_metrics = evaluate_predictions(val_out.y_true, val_out.y_prob, threshold_res.threshold)
        val_score = float(score_from_metrics(val_metrics, cfg.train.monitor_metric))

        msg = (
            f"[DURA-Stage3] epoch={epoch + 1} "
            f"train_loss={train_loss:.6f} cls={train_cls:.6f} orth={train_orth:.6f}"
        )
        if reliability_enabled:
            msg += f" rel={train_rel:.6f}"
        if domain_private_enabled:
            msg += f" dom_pr={train_domain_private:.6f}"
        msg += f" val_{cfg.train.monitor_metric}={val_score:.6f} best_threshold={threshold_res.threshold:.4f}"
        logger.info(msg)

        if tb_writer is not None:
            tb_writer.add_scalar("dura_stage3/train_loss", train_loss, epoch + 1)
            tb_writer.add_scalar("dura_stage3/train_cls_loss", train_cls, epoch + 1)
            tb_writer.add_scalar("dura_stage3/train_orth_loss", train_orth, epoch + 1)
            if reliability_enabled:
                tb_writer.add_scalar("dura_stage3/train_reliability_loss", train_rel, epoch + 1)
            if domain_private_enabled:
                tb_writer.add_scalar("dura_stage3/train_domain_private_loss", train_domain_private, epoch + 1)
            tb_writer.add_scalar("dura_stage3/val_macro_f1", float(val_metrics["overall"]["macro_f1"]), epoch + 1)
            tb_writer.add_scalar("dura_stage3/val_monitor_score", float(val_score), epoch + 1)

        improved = val_score > best_monitor if cfg.train.monitor_mode == "max" else val_score < best_monitor
        if improved:
            best_monitor = val_score
            best_threshold = float(threshold_res.threshold)
            best_val_metrics = val_metrics
            best_state = {
                "model": improved_state,
                "threshold": best_threshold,
                "val_score": best_monitor,
                "epoch": epoch + 1,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.early_stopping_patience:
                logger.info("[DURA-Stage3] early stop at epoch=%s", epoch + 1)
                break

    if best_state is None:
        best_state = {"model": _clone_state_dict(model), "threshold": best_threshold, "val_score": best_monitor, "epoch": 0}

    ckpt_path = out_dir / "dura_best.pt"
    save_checkpoint(ckpt_path, best_state)
    model.load_state_dict(best_state["model"], strict=True)
    val_out = evaluate_loader(model, val_loader, device)
    test_out = evaluate_loader(model, test_loader, device)
    val_metrics = evaluate_predictions(val_out.y_true, val_out.y_prob, best_threshold)
    test_metrics = evaluate_predictions(test_out.y_true, test_out.y_prob, best_threshold)
    per_domain = per_domain_metrics(test_out.y_true, test_out.y_prob, test_out.domains, threshold=best_threshold)
    write_metrics_report(out_dir / "val_report", {"threshold": best_threshold, **val_metrics})
    write_metrics_report(out_dir / "test_report", {"threshold": best_threshold, **test_metrics})
    write_per_domain_csv(out_dir / "test_per_domain.csv", per_domain)

    val_predictions_path = out_dir / "predictions_val.json"
    test_predictions_path = out_dir / "predictions_test.json"
    best_epoch = int(best_state.get("epoch", 0))
    _write_prediction_json(val_predictions_path, val_out, best_threshold, best_epoch)
    _write_prediction_json(test_predictions_path, test_out, best_threshold, best_epoch)
    return Stage3Result(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_score=float(best_monitor),
        val_metrics=best_val_metrics or val_metrics,
        test_metrics=test_metrics,
        per_domain=per_domain,
        checkpoint_path=ckpt_path,
        val_predictions_path=val_predictions_path,
        test_predictions_path=test_predictions_path,
    )


def build_class_weights(records_by_uid, train_ids, cfg: DURAConfig):
    return _compute_class_weights(records_by_uid, train_ids, cfg.loss.class_weighting)
