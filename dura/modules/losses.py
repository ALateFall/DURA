from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F



def _apply_class_weights(labels: torch.Tensor, class_weights: Optional[torch.Tensor]) -> torch.Tensor:
    if class_weights is None:
        return torch.ones_like(labels, dtype=torch.float32)
    cw = class_weights.to(labels.device)
    return cw[labels]



def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    kind: str,
    label_smoothing: float,
    focal_gamma: float,
    reduction: str = "mean",
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha_pos: float = 0.5,
) -> torch.Tensor:
    if logits.dim() == 1 or (logits.dim() == 2 and logits.size(-1) == 1):
        binary_logits = logits.reshape(-1)
        targets = labels.float()
        if kind == "ce":
            loss = F.binary_cross_entropy_with_logits(binary_logits, targets, reduction="none")
            weights = _apply_class_weights(labels, class_weights)
            loss = loss * weights
        elif kind == "focal":
            bce = F.binary_cross_entropy_with_logits(binary_logits, targets, reduction="none")
            prob = torch.sigmoid(binary_logits)
            pt = torch.where(labels == 1, prob, 1.0 - prob)
            alpha_t = torch.where(labels == 1, torch.full_like(bce, focal_alpha_pos), torch.full_like(bce, 1.0 - focal_alpha_pos))
            weights = _apply_class_weights(labels, class_weights)
            loss = alpha_t * ((1.0 - pt) ** focal_gamma) * bce * weights
        else:
            raise ValueError(f"Unsupported classification loss kind: {kind}")

        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        return loss.mean()

    if kind == "ce":
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
        weights = _apply_class_weights(labels, class_weights)
        loss = ce * weights

    elif kind == "focal":
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=label_smoothing)
        pt = torch.exp(-ce)
        alpha_t = torch.where(labels == 1, torch.full_like(ce, focal_alpha_pos), torch.full_like(ce, 1.0 - focal_alpha_pos))
        weights = _apply_class_weights(labels, class_weights)
        loss = alpha_t * ((1 - pt) ** focal_gamma) * ce * weights

    else:
        raise ValueError(f"Unsupported classification loss kind: {kind}")

    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()



def orthogonality_loss(z_shared: torch.Tensor, z_private: torch.Tensor) -> torch.Tensor:
    s = F.normalize(z_shared, p=2, dim=-1)
    p = F.normalize(z_private, p=2, dim=-1)
    cos_sim = torch.sum(s * p, dim=-1)
    return torch.mean(torch.abs(cos_sim))
