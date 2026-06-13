from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from dura.data_utils.cleaning import NewsRecord
from dura.common.io import write_json, write_text


@dataclass
class OverlapReport:
    train_val_overlap: int
    train_test_overlap: int
    val_test_overlap: int
    has_overlap: bool


def check_overlap(train_ids: Iterable[str], val_ids: Iterable[str], test_ids: Iterable[str]) -> OverlapReport:
    t = set(train_ids)
    v = set(val_ids)
    s = set(test_ids)
    tv = len(t & v)
    ts = len(t & s)
    vs = len(v & s)
    return OverlapReport(
        train_val_overlap=tv,
        train_test_overlap=ts,
        val_test_overlap=vs,
        has_overlap=(tv + ts + vs) > 0,
    )

def split_statistics(records_by_uid: Dict[str, NewsRecord], split_ids: List[str]) -> Dict[str, float]:
    if not split_ids:
        return {
            "n": 0,
            "fake": 0,
            "real": 0,
            "missing_image": 0,
            "missing_image_ratio": 0.0,
        }

    n = len(split_ids)
    fake = sum(1 for uid in split_ids if records_by_uid[uid].label == 1)
    missing = sum(1 for uid in split_ids if records_by_uid[uid].has_image == 0)

    return {
        "n": n,
        "fake": fake,
        "real": n - fake,
        "missing_image": missing,
        "missing_image_ratio": missing / max(1, n),
    }



def domain_statistics(records: List[NewsRecord]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[NewsRecord]] = {}
    for r in records:
        buckets.setdefault(r.domain, []).append(r)

    out: Dict[str, Dict[str, float]] = {}
    for domain, rs in sorted(buckets.items()):
        n = len(rs)
        fake = sum(1 for x in rs if x.label == 1)
        missing = sum(1 for x in rs if x.has_image == 0)
        out[domain] = {
            "n": n,
            "fake": fake,
            "real": n - fake,
            "fake_ratio": fake / max(1, n),
            "missing_image": missing,
            "missing_image_ratio": missing / max(1, n),
        }
    return out



def save_audit_report(
    out_dir: Path,
    overlap: OverlapReport,
    split_stats: Dict[str, Dict[str, float]],
    domain_stats: Dict[str, Dict[str, float]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "overlap": asdict(overlap),
        "split_stats": split_stats,
        "domain_stats": domain_stats,
    }
    write_json(out_dir / "audit_report.json", payload)

    lines = [
        "# Data Audit",
        "",
        "## Overlap",
        f"- train/val overlap: {overlap.train_val_overlap}",
        f"- train/test overlap: {overlap.train_test_overlap}",
        f"- val/test overlap: {overlap.val_test_overlap}",
        f"- has_overlap: {overlap.has_overlap}",
        "",
        "## Split Stats",
    ]

    for split_name, stats in split_stats.items():
        lines.append(
            f"- {split_name}: n={stats['n']} fake={stats['fake']} real={stats['real']} "
            f"missing_image={stats['missing_image']} missing_image_ratio={stats['missing_image_ratio']:.4f}"
        )

    lines.extend(["", "## Domain Stats"])
    for domain, stats in domain_stats.items():
        lines.append(
            f"- {domain}: n={stats['n']} fake_ratio={stats['fake_ratio']:.4f} "
            f"missing_image_ratio={stats['missing_image_ratio']:.4f}"
        )

    write_text(out_dir / "audit_report.md", "\n".join(lines))
