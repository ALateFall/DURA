#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Tuple


METRICS = ["acc", "macro_f1", "real_f1", "fake_f1"]
MISSING_ORDER = ["Clean", "w/o Image", "w/o Text", "Image Missing 50%", "Text Missing 50%"]
NOISE_ORDER = ["Clean", "+1 Noise Image", "+2 Noise Images", "+4 Noise Images"]
NOISE_AGG_ORDER = ["Clean", "+1 Noise Image", "+2 Noise Images", "+4 Noise Images", "Avg. Noise", "Worst Noise"]
MODEL_ORDER = ["DURA", "w/o IVW", "w/o Adaptive Fusion"]


def parse_args():
    p = argparse.ArgumentParser(description="Summarize DURA robustness runs into mean/std tables.")
    p.add_argument("--run-dir", action="append", default=[], help="Robustness run directory. Can be repeated.")
    p.add_argument("--run-glob", action="append", default=[], help="Glob pattern for robustness run directories.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--dataset-note", type=str, default="")
    return p.parse_args()


def _read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float, std: float) -> str:
    return f"{value:.4f}+/-{std:.4f}"


def _fmt_latexish(value: float, std: float) -> str:
    return f"{value:.4f}+-{std:.4f}"


def _model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return MODEL_ORDER.index(model), model
    except ValueError:
        return len(MODEL_ORDER), model


def _condition_order(table: str) -> List[str]:
    if table == "missing_modality":
        return MISSING_ORDER
    if table == "image_noise":
        return NOISE_ORDER
    if table == "image_noise_aggregate":
        return NOISE_AGG_ORDER
    return []


def _collect_run_dirs(args) -> List[Path]:
    dirs: List[Path] = []
    for raw in args.run_dir:
        dirs.append(Path(raw))
    for pat in args.run_glob:
        dirs.extend(Path().glob(pat))
    unique: Dict[str, Path] = {}
    for d in dirs:
        d = d.resolve()
        if (d / "summary.json").exists():
            unique[str(d)] = d
    return [unique[k] for k in sorted(unique)]


def _load_rows(run_dirs: Iterable[Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run_dir in run_dirs:
        payload = _read_json(run_dir / "summary.json")
        seed = int(payload.get("seed", -1))
        checkpoint_epoch = int(payload.get("checkpoint_epoch", 0))
        source_checkpoint = str(payload.get("source_checkpoint", ""))
        for row in payload.get("rows", []):
            out = {
                "seed": seed,
                "model": str(row["model"]),
                "table": str(row["table"]),
                "condition": str(row["condition"]),
                "checkpoint_epoch": checkpoint_epoch,
                "threshold": float(row["threshold"]),
                "source_checkpoint": source_checkpoint,
                "run_dir": str(run_dir),
            }
            for metric in METRICS:
                out[metric] = float(row[metric])
            rows.append(out)
    return rows


def _add_noise_aggregate_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = list(rows)
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["table"] == "image_noise" and row["condition"] != "Clean":
            grouped[(int(row["seed"]), str(row["model"]))].append(row)

    for (_seed, _model), group_rows in grouped.items():
        by_cond = {str(r["condition"]): r for r in group_rows}
        if not all(cond in by_cond for cond in NOISE_ORDER[1:]):
            continue
        template = by_cond[NOISE_ORDER[1]]
        for agg_name, reducer in [
            ("Avg. Noise", lambda vals: mean(vals)),
            ("Worst Noise", lambda vals: min(vals)),
        ]:
            agg = {
                "seed": template["seed"],
                "model": template["model"],
                "table": "image_noise_aggregate",
                "condition": agg_name,
                "checkpoint_epoch": template["checkpoint_epoch"],
                "threshold": template["threshold"],
                "source_checkpoint": template["source_checkpoint"],
                "run_dir": template["run_dir"],
            }
            for metric in METRICS:
                agg[metric] = float(reducer([float(by_cond[c][metric]) for c in NOISE_ORDER[1:]]))
            out.append(agg)
        clean = {
            **{k: template[k] for k in ["seed", "model", "checkpoint_epoch", "threshold", "source_checkpoint", "run_dir"]},
            "table": "image_noise_aggregate",
            "condition": "Clean",
        }
        clean_src = next(r for r in rows if r["table"] == "image_noise" and r["condition"] == "Clean" and r["seed"] == template["seed"] and r["model"] == template["model"])
        for metric in METRICS:
            clean[metric] = float(clean_src[metric])
        out.append(clean)
        for cond in NOISE_ORDER[1:]:
            src = by_cond[cond]
            copied = dict(src)
            copied["table"] = "image_noise_aggregate"
            out.append(copied)
    return out


def _summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["table"]), str(row["condition"]))].append(row)

    summary: List[Dict[str, object]] = []
    for (model, table, condition), group_rows in grouped.items():
        item: Dict[str, object] = {
            "model": model,
            "table": table,
            "condition": condition,
            "n": len(group_rows),
            "checkpoint_epoch": int(group_rows[0]["checkpoint_epoch"]),
            "threshold": float(group_rows[0]["threshold"]),
        }
        for metric in METRICS:
            vals = [float(r[metric]) for r in group_rows]
            item[f"{metric}_mean"] = mean(vals)
            item[f"{metric}_std"] = stdev(vals) if len(vals) > 1 else 0.0
        summary.append(item)

    def key(row):
        order = _condition_order(str(row["table"]))
        cond = str(row["condition"])
        try:
            cidx = order.index(cond)
        except ValueError:
            cidx = len(order)
        return str(row["table"]), _model_sort_key(str(row["model"])), cidx, cond

    return sorted(summary, key=key)


def _write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_matrix_csv(out_dir: Path, summary: List[Dict[str, object]], table: str, metric: str, conditions: List[str]) -> None:
    models = sorted({str(r["model"]) for r in summary if r["table"] == table}, key=_model_sort_key)
    by_key = {(str(r["model"]), str(r["condition"])): r for r in summary if r["table"] == table}
    path = out_dir / f"{table}_{metric}_mean_std_matrix.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", *conditions])
        for model in models:
            row = [model]
            for cond in conditions:
                item = by_key.get((model, cond))
                if item is None:
                    row.append("")
                else:
                    row.append(_fmt(float(item[f"{metric}_mean"]), float(item[f"{metric}_std"])))
            writer.writerow(row)


def _markdown_table(summary: List[Dict[str, object]], table: str, title: str, conditions: List[str]) -> str:
    rows = [f"## {title}", ""]
    rows.append("| Model | " + " | ".join(conditions) + " |")
    rows.append("| --- | " + " | ".join(["---"] * len(conditions)) + " |")
    models = sorted({str(r["model"]) for r in summary if r["table"] == table}, key=_model_sort_key)
    by_key = {(str(r["model"]), str(r["condition"])): r for r in summary if r["table"] == table}
    for model in models:
        cells = [model]
        for cond in conditions:
            item = by_key.get((model, cond))
            if item is None:
                cells.append("")
                continue
            acc = _fmt_latexish(float(item["acc_mean"]), float(item["acc_std"]))
            f1 = _fmt_latexish(float(item["macro_f1_mean"]), float(item["macro_f1_std"]))
            cells.append(f"{acc} / {f1}")
        rows.append("| " + " | ".join(cells) + " |")
    rows.append("")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    run_dirs = _collect_run_dirs(args)
    if not run_dirs:
        raise SystemExit("No run directories with summary.json found.")

    actual_rows = _load_rows(run_dirs)
    rows = _add_noise_aggregate_rows(actual_rows)
    summary = _summarize(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_fields = ["seed", "model", "table", "condition", "checkpoint_epoch", "threshold", *METRICS, "source_checkpoint", "run_dir"]
    _write_csv(out_dir / "all_seed_rows.csv", actual_rows, all_fields)
    _write_csv(out_dir / "all_seed_rows_with_noise_aggregates.csv", rows, all_fields)

    summary_fields = ["model", "table", "condition", "n", "checkpoint_epoch", "threshold"]
    for metric in METRICS:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    _write_csv(out_dir / "mean_std_rows.csv", summary, summary_fields)

    for metric in METRICS:
        _write_matrix_csv(out_dir, summary, "missing_modality", metric, MISSING_ORDER)
        _write_matrix_csv(out_dir, summary, "image_noise", metric, NOISE_ORDER)
        _write_matrix_csv(out_dir, summary, "image_noise_aggregate", metric, NOISE_AGG_ORDER)

    md = [
        "# DURA Robustness Results",
        "",
        args.dataset_note or "Dataset: configured evaluation split.",
        f"Runs: {len(run_dirs)}",
        "Cell format: Accuracy / Macro-F1. Values are mean+/-std.",
        "",
        _markdown_table(summary, "missing_modality", "Missing Modality Robustness", MISSING_ORDER),
        _markdown_table(summary, "image_noise", "Nested Noisy Image Injection Robustness", NOISE_ORDER),
        _markdown_table(summary, "image_noise_aggregate", "Nested Noisy Image Injection Robustness with Aggregates", NOISE_AGG_ORDER),
    ]
    (out_dir / "robustness_tables_mean_std.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "runs": len(run_dirs)}, indent=2))


if __name__ == "__main__":
    main()
