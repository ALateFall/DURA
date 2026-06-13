from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import torch

from dura.common.checkpoint import save_checkpoint


def _dare_sparse_rescale(t: torch.Tensor, drop_rate: float, generator: torch.Generator) -> torch.Tensor:
    if drop_rate <= 0:
        return t
    keep_prob = 1.0 - drop_rate
    mask = (torch.rand(t.shape, generator=generator, device=t.device) < keep_prob).to(t.dtype)
    return (t * mask) / max(keep_prob, 1e-8)


def _weighted_sum(
    source_adapter_states: Dict[str, Dict[str, torch.Tensor]],
    domains: List[str],
    keys: List[str],
    lambdas: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = torch.zeros_like(source_adapter_states[domains[0]][k], dtype=torch.float32)
        for d in domains:
            w = source_adapter_states[d][k].detach().cpu().float()
            acc = acc + lambdas[d] * w
        merged[k] = acc
    return merged


def _weighted_dare(
    cfg: Any,
    source_adapter_states: Dict[str, Dict[str, torch.Tensor]],
    domains: List[str],
    keys: List[str],
    lambdas: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(cfg.dare.seed)

    merged: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = torch.zeros_like(source_adapter_states[domains[0]][k], dtype=torch.float32)
        for d in domains:
            w = source_adapter_states[d][k].detach().cpu().float()
            sparse = _dare_sparse_rescale(w, drop_rate=cfg.dare.drop_rate, generator=gen)
            acc = acc + lambdas[d] * sparse
        merged[k] = acc
    return merged


def _sign_agreement_fraction(stacked: torch.Tensor) -> torch.Tensor:
    # stacked: [n_domains, ...]
    pos = (stacked > 0).float().mean(dim=0)
    neg = (stacked < 0).float().mean(dim=0)
    return torch.maximum(pos, neg)


def _weighted_ties(
    source_adapter_states: Dict[str, Dict[str, torch.Tensor]],
    domains: List[str],
    keys: List[str],
    lambdas: Dict[str, float],
    threshold: float = 0.6,
) -> Dict[str, torch.Tensor]:
    # Lightweight TIES-like merge:
    # 1) compute weighted average
    # 2) keep coordinates where sign agreement across domains is high enough
    # 3) zero-out low-agreement coordinates
    merged: Dict[str, torch.Tensor] = {}
    lambda_tensor = torch.tensor([lambdas[d] for d in domains], dtype=torch.float32)
    for k in keys:
        tensors = [source_adapter_states[d][k].detach().cpu().float() for d in domains]
        stacked = torch.stack(tensors, dim=0)

        avg = (stacked * lambda_tensor.view(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)

        agree = _sign_agreement_fraction(stacked)
        mask = (agree >= threshold).to(avg.dtype)

        merged[k] = avg * mask
    return merged


def merge_source_adapters_with_dare(
    cfg: Any,
    source_adapter_states: Dict[str, Dict[str, torch.Tensor]],
    out_dir: Path,
    logger,
    domain_sizes: Dict[str, int] | None = None,
) -> Dict[str, torch.Tensor]:
    domains = sorted(source_adapter_states.keys())
    if not domains:
        raise RuntimeError("No source adapters to merge")

    keys = sorted(source_adapter_states[domains[0]].keys())
    for d in domains[1:]:
        if sorted(source_adapter_states[d].keys()) != keys:
            raise RuntimeError(f"Adapter key mismatch for domain {d}")

    if cfg.dare.merge_weights == "equal":
        lambdas = {d: 1.0 / len(domains) for d in domains}
    elif cfg.dare.merge_weights == "domain_size":
        if not domain_sizes:
            raise RuntimeError("dare.merge_weights=domain_size requires domain_sizes")
        total = float(sum(domain_sizes.get(d, 0) for d in domains))
        if total <= 0:
            raise RuntimeError("domain_sizes sum is zero")
        lambdas = {d: float(domain_sizes.get(d, 0)) / total for d in domains}
    else:
        raise ValueError(f"Unsupported dare.merge_weights={cfg.dare.merge_weights}")

    method = cfg.dare.method
    if method == "dare":
        merged = _weighted_dare(cfg, source_adapter_states, domains, keys, lambdas)
    elif method == "avg":
        merged = _weighted_sum(source_adapter_states, domains, keys, lambdas)
    elif method == "ties":
        merged = _weighted_ties(source_adapter_states, domains, keys, lambdas, threshold=0.6)
    else:
        raise ValueError(f"Unsupported dare.method={method}")

    save_checkpoint(
        out_dir / "stage2_shared_adapter.pt",
        {
            "domains": domains,
            "method": method,
            "drop_rate": cfg.dare.drop_rate,
            "merge_weights": cfg.dare.merge_weights,
            "lambdas": lambdas,
            "shared_adapter_state": merged,
        },
    )

    logger.info(
        "[Stage2] merged %s domain adapters with method=%s drop_rate=%.3f merge_weights=%s",
        len(domains),
        method,
        cfg.dare.drop_rate,
        cfg.dare.merge_weights,
    )
    return merged
