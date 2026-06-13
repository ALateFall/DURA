from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from dura.eval_utils.metrics import evaluate_predictions


@dataclass
class ThresholdSearchResult:
    threshold: float
    score: float
    metric_name: str


def score_from_metrics(metrics: dict, metric_name: str) -> float:
    overall = metrics["overall"]
    per_class = metrics["per_class"]
    if metric_name == "real_f1":
        return float(per_class["real"]["f1"])
    if metric_name == "fake_f1":
        return float(per_class["fake"]["f1"])
    if metric_name == "acc_macro_f1":
        return 0.5 * float(overall["acc"]) + 0.5 * float(overall["macro_f1"])
    return float(overall[metric_name])



def find_best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    metric_name: str,
    t_min: float,
    t_max: float,
    t_step: float,
) -> ThresholdSearchResult:
    best_t = t_min
    best_score = -1.0

    t = t_min
    while t <= t_max + 1e-8:
        m = evaluate_predictions(y_true=y_true, y_prob=y_prob, threshold=t)
        score = score_from_metrics(m, metric_name)

        if score > best_score:
            best_score = score
            best_t = t
        t += t_step

    return ThresholdSearchResult(threshold=float(best_t), score=float(best_score), metric_name=metric_name)
