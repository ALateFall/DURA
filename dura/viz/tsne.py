from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dura.common.io import write_json


def _setup_tsne_style() -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False


def _domain_colors(n: int) -> List[str]:
    if n <= 2:
        return ["#0072B2", "#D55E00"][:n]
    base = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#F0E442",
        "#1F77B4",
        "#D62728",
        "#2CA02C",
    ]
    return [base[i % len(base)] for i in range(n)]


def _domain_legend_label(local_rank: int) -> str:
    return f"Domain {local_rank}"


def _standardize_features(features: np.ndarray) -> np.ndarray:
    x = features.astype(np.float32)
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    return (x - mu) / std


def _reduce_features(features: np.ndarray, pre_pca_dim: int) -> np.ndarray:
    if pre_pca_dim <= 0 or features.shape[1] <= pre_pca_dim:
        return features
    pca = PCA(n_components=min(pre_pca_dim, features.shape[1]), random_state=42)
    return pca.fit_transform(features)


def _fit_tsne(features: np.ndarray, perplexity: int, random_state: int) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def _sample_random_indices(n: int, max_points: int, random_state: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _sample_balanced_domain_indices(domains: List[str], max_per_domain: int, random_state: int) -> np.ndarray:
    bucket: Dict[str, List[int]] = defaultdict(list)
    for i, d in enumerate(domains):
        bucket[d].append(i)

    rng = np.random.default_rng(random_state)
    kept: List[int] = []
    for d in sorted(bucket.keys()):
        arr = np.asarray(bucket[d], dtype=np.int64)
        if arr.size > max_per_domain:
            sel = rng.choice(arr, size=max_per_domain, replace=False)
            sel.sort()
            kept.extend(sel.tolist())
        else:
            kept.extend(arr.tolist())

    kept = sorted(kept)
    return np.asarray(kept, dtype=np.int64)


def _filter_by_focus_domains(
    features: np.ndarray,
    labels: List[int],
    domains: List[str],
    focus_domains: Sequence[str] | None,
) -> tuple[np.ndarray, List[int], List[str], List[str]]:
    if not focus_domains:
        return features, labels, domains, []

    focus_set = set(focus_domains)
    kept_idx = [i for i, d in enumerate(domains) if d in focus_set]
    if not kept_idx:
        raise ValueError(f"No samples match focus_domains={list(focus_domains)}")

    missing = sorted(focus_set - set(domains))
    features = features[np.asarray(kept_idx, dtype=np.int64)]
    labels = [labels[i] for i in kept_idx]
    domains = [domains[i] for i in kept_idx]
    return features, labels, domains, missing


def _plot_by_label(z: np.ndarray, labels: List[int], feature_type: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 6), facecolor="#FFFFFF")
    ax = plt.gca()
    ax.set_facecolor("#FFFFFF")

    arr_labels = np.asarray(labels)
    cmap = plt.get_cmap("viridis")

    for cls in [0, 1]:
        mask = arr_labels == cls
        if not np.any(mask):
            continue
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=120,
            alpha=0.6,
            c=[cmap(float(cls))],
            edgecolors="none",
            linewidths=0.0,
            label=("real" if cls == 0 else "fake"),
        )

    plt.title(f"t-SNE Visualization ({feature_type}) by Label", fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_by_domain(
    z: np.ndarray,
    domains: List[str],
    feature_type: str,
    out_path: Path,
    full_out_path: Path,
    random_state: int,
    max_points: int,
    show_centers: bool = False,
) -> Dict[str, object]:
    present_domains = sorted(set(domains))
    arr_domains = np.asarray(domains)
    colors = _domain_colors(len(present_domains))
    legend_cols = max(1, min(3, len(present_domains)))

    plt.figure(figsize=(8, 6), facecolor="#FFFFFF")
    ax = plt.gca()
    ax.set_facecolor("#FFFFFF")

    for i, domain_name in enumerate(present_domains, start=1):
        mask = arr_domains == domain_name
        if not np.any(mask):
            continue
        plt.scatter(
            z[mask, 0],
            z[mask, 1],
            s=120,
            alpha=0.6,
            c=[colors[i - 1]],
            edgecolors="none",
            linewidths=0.0,
            label=_domain_legend_label(i),
        )

    plt.title(f"t-SNE Visualization ({feature_type}) by Domain", fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(markerscale=1.20, fontsize=9, ncol=legend_cols, frameon=True)
    plt.tight_layout()
    plt.savefig(full_out_path, dpi=220)
    plt.close()

    max_per_domain = max(40, min(260, max_points // max(1, len(present_domains))))
    idx_bal = _sample_balanced_domain_indices(
        domains=domains,
        max_per_domain=max_per_domain,
        random_state=random_state + 17,
    )
    z_bal = z[idx_bal]
    domains_bal = [domains[i] for i in idx_bal.tolist()]
    arr_domains_bal = np.asarray(domains_bal)

    plt.figure(figsize=(8, 6), facecolor="#FFFFFF")
    ax = plt.gca()
    ax.set_facecolor("#FFFFFF")

    domain_index_map: Dict[str, str] = {}

    for i, domain_name in enumerate(present_domains, start=1):
        mask = arr_domains_bal == domain_name
        if not np.any(mask):
            continue

        c = colors[i - 1]
        legend_label = _domain_legend_label(i)
        plt.scatter(
            z_bal[mask, 0],
            z_bal[mask, 1],
            s=120,
            alpha=0.6,
            c=[c],
            edgecolors="none",
            linewidths=0.0,
            label=legend_label,
        )

        if show_centers:
            center = z_bal[mask].mean(axis=0)
            plt.scatter(
                [center[0]],
                [center[1]],
                s=100,
                marker="X",
                c=[c],
                edgecolors="#111111",
                linewidths=0.45,
                zorder=5,
            )
            plt.text(
                center[0],
                center[1],
                legend_label,
                fontsize=8,
                color="#111111",
                ha="center",
                va="center",
            )

        domain_index_map[legend_label] = domain_name

    plt.title(f"t-SNE Visualization ({feature_type}) by Domain", fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(markerscale=1.20, fontsize=9, ncol=legend_cols, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "max_per_domain": int(max_per_domain),
        "n_balanced_points": int(z_bal.shape[0]),
        "domain_index_map": domain_index_map,
        "primary_plot": str(out_path.name),
        "full_plot": str(full_out_path.name),
        "show_centers": bool(show_centers),
    }


def run_tsne(
    features: np.ndarray,
    labels: List[int],
    domains: List[str],
    feature_type: str,
    out_dir: Path,
    perplexity: int,
    random_state: int,
    max_points: int,
    focus_domains: Sequence[str] | None = None,
    show_domain_centers: bool = False,
    pre_pca_dim: int = 50,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_tsne_style()

    n_raw_total = int(features.shape[0])

    features, labels, domains, missing_focus_domains = _filter_by_focus_domains(
        features=features,
        labels=labels,
        domains=domains,
        focus_domains=focus_domains,
    )

    n_focus_raw = int(features.shape[0])
    idx = _sample_random_indices(n=n_focus_raw, max_points=max_points, random_state=random_state)

    features = features[idx]
    labels = [labels[i] for i in idx.tolist()]
    domains = [domains[i] for i in idx.tolist()]

    standardized = _standardize_features(features)
    reduced = _reduce_features(standardized, pre_pca_dim=pre_pca_dim)
    z = _fit_tsne(features=reduced, perplexity=perplexity, random_state=random_state)

    label_out = out_dir / f"tsne_{feature_type}_by_label.png"
    domain_out = out_dir / f"tsne_{feature_type}_by_domain.png"
    domain_full_out = out_dir / f"tsne_{feature_type}_by_domain_full.png"

    _plot_by_label(z=z, labels=labels, feature_type=feature_type, out_path=label_out)
    domain_meta = _plot_by_domain(
        z=z,
        domains=domains,
        feature_type=feature_type,
        out_path=domain_out,
        full_out_path=domain_full_out,
        random_state=random_state,
        max_points=max_points,
        show_centers=show_domain_centers,
    )

    metadata = {
        "feature_type": feature_type,
        "n_points_raw_total": n_raw_total,
        "n_points_after_focus": n_focus_raw,
        "n_points_used": int(features.shape[0]),
        "perplexity": perplexity,
        "random_state": random_state,
        "pre_pca_dim": int(pre_pca_dim),
        "sampling": {
            "raw_to_used": "random_subsample_if_needed",
            "max_points": int(max_points),
            "domain_balanced_for_primary_domain_plot": True,
            "focus_domains": list(focus_domains) if focus_domains else [],
            "missing_focus_domains": missing_focus_domains,
            **domain_meta,
        },
        "style": {
            "font": "Times New Roman (fallback: Times, DejaVu Serif)",
            "point_size": {
                "label_plot": 120,
                "domain_plot_balanced": 120,
                "domain_plot_full": 120,
            },
            "alpha": {
                "label_plot": 0.6,
                "domain_plot_balanced": 0.6,
                "domain_plot_full": 0.6,
            },
            "background": "#FFFFFF",
        },
        "note": "t-SNE is for representation structure observation only, not a substitute for classification metrics.",
    }
    write_json(out_dir / f"tsne_{feature_type}_meta.json", metadata)

