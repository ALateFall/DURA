from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class MetricResult:
    acc: float
    precision: float
    recall: float
    f1: float


def binary_auc(y_true: np.ndarray, y_prob: np.ndarray, positive_label: int = 1) -> float:
    assert y_true.shape == y_prob.shape
    y_bin = (y_true == positive_label).astype(np.int64)
    n_pos = int(y_bin.sum())
    n_neg = int(y_bin.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_prob, kind="mergesort")
    probs_sorted = y_prob[order]
    ranks = np.zeros_like(y_prob, dtype=np.float64)
    i = 0
    n = probs_sorted.size
    while i < n:
        j = i + 1
        while j < n and probs_sorted[j] == probs_sorted[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    rank_sum_pos = float(ranks[y_bin == 1].sum())
    auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)



def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0



def confusion_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)



def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, positive_label: int = 1) -> MetricResult:
    assert y_true.shape == y_pred.shape

    if positive_label == 1:
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
    else:
        tp = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 1) & (y_pred == 0)).sum())
        fn = int(((y_true == 0) & (y_pred == 1)).sum())

    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return MetricResult(acc=acc, precision=precision, recall=recall, f1=f1)



def class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    real = binary_metrics(y_true, y_pred, positive_label=0)
    fake = binary_metrics(y_true, y_pred, positive_label=1)

    real_acc = _safe_div(float(((y_true == 0) & (y_pred == 0)).sum()), float((y_true == 0).sum()))
    fake_acc = _safe_div(float(((y_true == 1) & (y_pred == 1)).sum()), float((y_true == 1).sum()))

    return {
        "real": {
            "acc": real_acc,
            "precision": real.precision,
            "recall": real.recall,
            "f1": real.f1,
        },
        "fake": {
            "acc": fake_acc,
            "precision": fake.precision,
            "recall": fake.recall,
            "f1": fake.f1,
        },
    }



def evaluate_predictions(y_true: Sequence[int], y_prob: Sequence[float], threshold: float) -> Dict[str, object]:
    yt = np.asarray(y_true, dtype=np.int64)
    yp_prob = np.asarray(y_prob, dtype=np.float64)
    yp = (yp_prob >= threshold).astype(np.int64)

    overall = binary_metrics(yt, yp, positive_label=1)
    auc = binary_auc(yt, yp_prob, positive_label=1)
    per_class = class_metrics(yt, yp)
    cm = confusion_from_preds(yt, yp)

    macro_f1 = 0.5 * (per_class["real"]["f1"] + per_class["fake"]["f1"])
    balanced_acc = 0.5 * (per_class["real"]["acc"] + per_class["fake"]["acc"])

    return {
        "threshold": float(threshold),
        "overall": {
            "acc": overall.acc,
            "precision": overall.precision,
            "recall": overall.recall,
            "f1": overall.f1,
            "auc": float(auc),
            "macro_f1": float(macro_f1),
            "balanced_acc": float(balanced_acc),
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }



def per_domain_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    domains: Sequence[str],
    threshold: float,
) -> Dict[str, Dict[str, object]]:
    yt = np.asarray(y_true, dtype=np.int64)
    yp_prob = np.asarray(y_prob, dtype=np.float64)
    yp = (yp_prob >= threshold).astype(np.int64)
    domains = np.asarray(domains)

    out: Dict[str, Dict[str, object]] = {}
    for d in sorted(set(domains.tolist())):
        mask = domains == d
        d_true = yt[mask]
        d_prob = yp_prob[mask]

        metric = evaluate_predictions(d_true.tolist(), d_prob.tolist(), threshold)
        out[d] = {
            "n": int(mask.sum()),
            "metrics": metric,
        }
    return out
