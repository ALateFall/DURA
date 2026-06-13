from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, List

from dura.common.io import write_json, write_text


_EVENT_RE = re.compile(r"^event_(\d+)$")


def _domain_sort_key(domain: str):
    m = _EVENT_RE.match(str(domain))
    if m:
        return (0, int(m.group(1)))
    return (1, str(domain))



def write_metrics_report(out_dir: Path, payload: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "metrics.json", payload)

    overall = payload.get("overall", {})
    per_class = payload.get("per_class", {})
    cm = payload.get("confusion_matrix", [[0, 0], [0, 0]])

    lines = [
        "# Evaluation Report",
        "",
        f"- threshold: {payload.get('threshold', 0.5):.4f}",
        f"- acc: {overall.get('acc', 0.0):.4f}",
        f"- precision: {overall.get('precision', 0.0):.4f}",
        f"- recall: {overall.get('recall', 0.0):.4f}",
        f"- f1: {overall.get('f1', 0.0):.4f}",
        f"- auc: {overall.get('auc', 0.0):.4f}",
        f"- macro_f1: {overall.get('macro_f1', 0.0):.4f}",
        f"- balanced_acc: {overall.get('balanced_acc', 0.0):.4f}",
        "",
        "## Per-class",
        (
            f"- real: acc={per_class.get('real', {}).get('acc', 0.0):.4f} "
            f"precision={per_class.get('real', {}).get('precision', 0.0):.4f} "
            f"recall={per_class.get('real', {}).get('recall', 0.0):.4f} "
            f"f1={per_class.get('real', {}).get('f1', 0.0):.4f}"
        ),
        (
            f"- fake: acc={per_class.get('fake', {}).get('acc', 0.0):.4f} "
            f"precision={per_class.get('fake', {}).get('precision', 0.0):.4f} "
            f"recall={per_class.get('fake', {}).get('recall', 0.0):.4f} "
            f"f1={per_class.get('fake', {}).get('f1', 0.0):.4f}"
        ),
        "",
        "## Confusion Matrix",
        "- rows=true [real,fake], cols=pred [real,fake]",
        f"- [{cm[0][0]}, {cm[0][1]}]",
        f"- [{cm[1][0]}, {cm[1][1]}]",
    ]

    write_text(out_dir / "metrics.md", "\n".join(lines))



def write_per_domain_csv(out_path: Path, per_domain: Dict[str, Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "domain",
            "n",
            "acc",
            "precision",
            "recall",
            "f1",
            "auc",
            "macro_f1",
            "balanced_acc",
            "real_acc",
            "real_precision",
            "real_recall",
            "real_f1",
            "fake_acc",
            "fake_precision",
            "fake_recall",
            "fake_f1",
        ])
        for domain, obj in sorted(per_domain.items(), key=lambda x: _domain_sort_key(x[0])):
            n = obj["n"]
            metric = obj["metrics"]
            ov = metric["overall"]
            pc = metric["per_class"]
            writer.writerow(
                [
                    domain,
                    n,
                    ov["acc"],
                    ov["precision"],
                    ov["recall"],
                    ov["f1"],
                    ov.get("auc", 0.0),
                    ov.get("macro_f1", 0.0),
                    ov.get("balanced_acc", 0.0),
                    pc["real"]["acc"],
                    pc["real"]["precision"],
                    pc["real"]["recall"],
                    pc["real"]["f1"],
                    pc["fake"]["acc"],
                    pc["fake"]["precision"],
                    pc["fake"]["recall"],
                    pc["fake"]["f1"],
                ]
            )



def write_batch_summary(
    out_dir: Path,
    rows: List[Dict[str, object]],
    metric_key: str = "f1",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "summary.json", {"rows": rows})

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_name", "run_dir", "acc", "precision", "recall", "f1", "macro_f1", "balanced_acc"])
        for row in rows:
            writer.writerow(
                [
                    row.get("run_name", row.get("name", "")),
                    row.get("run_dir", ""),
                    row.get("acc", 0.0),
                    row.get("precision", 0.0),
                    row.get("recall", 0.0),
                    row.get("f1", 0.0),
                    row.get("macro_f1", 0.0),
                    row.get("balanced_acc", 0.0),
                ]
            )

    vals = [float(r.get(metric_key, 0.0)) for r in rows]
    if vals:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        best = max(rows, key=lambda x: float(x.get(metric_key, 0.0)))
        worst = min(rows, key=lambda x: float(x.get(metric_key, 0.0)))
    else:
        mean = var = 0.0
        best = worst = {}

    stats = {
        "metric": metric_key,
        "mean": mean,
        "variance": var,
        "best": best,
        "worst": worst,
    }
    write_json(out_dir / "aggregate_stats.json", stats)

    md = [
        "# Batch Summary",
        "",
        f"- metric: {metric_key}",
        f"- mean: {mean:.4f}",
        f"- variance: {var:.6f}",
        f"- best: {best.get('run_name', best.get('name', ''))} ({best.get(metric_key, 0.0):.4f})",
        f"- worst: {worst.get('run_name', worst.get('name', ''))} ({worst.get(metric_key, 0.0):.4f})",
    ]
    write_text(out_dir / "aggregate_stats.md", "\n".join(md))
