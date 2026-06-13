from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import random

from dura.data_utils.cleaning import NewsRecord
from dura.common.io import write_json


@dataclass
class SplitResult:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]
    split_map_path: Path



def _shuffle_inplace(items: List[str], seed: int) -> None:
    rnd = random.Random(seed)
    rnd.shuffle(items)



def _build_id_map(records: List[NewsRecord]) -> Dict[str, NewsRecord]:
    return {r.uid: r for r in records}



def _split_ids(ids: List[str], val_ratio: float, test_ratio: float, seed: int) -> tuple[List[str], List[str], List[str]]:
    ids = list(ids)
    _shuffle_inplace(ids, seed)
    n = len(ids)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_ids = ids[:n_test]
    val_ids = ids[n_test : n_test + n_val]
    train_ids = ids[n_test + n_val :]
    return train_ids, val_ids, test_ids



def build_splits(
    records: List[NewsRecord],
    mode: str,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_map_path: Path,
) -> SplitResult:
    id_map = _build_id_map(records)
    all_ids = sorted(id_map.keys())

    if mode != "all-domain":
        raise ValueError(f"Unsupported mode: {mode}")
    train_ids, val_ids, test_ids = _split_ids(all_ids, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    payload = {
        "mode": mode,
        "seed": seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }

    split_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(split_map_path, payload)

    return SplitResult(
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        test_ids=sorted(test_ids),
        split_map_path=split_map_path,
    )



def _read_uid_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                lines.append(x)
    return lines


def build_splits_from_predefined_files(
    split_dir: Path,
    split_map_path: Path,
) -> SplitResult:
    train_ids = _read_uid_lines(split_dir / "train_ids.txt")
    val_ids = _read_uid_lines(split_dir / "val_ids.txt")
    test_ids = _read_uid_lines(split_dir / "test_ids.txt")

    payload = {
        "mode": "predefined",
        "seed": -1,
        "val_ratio": 0.0,
        "test_ratio": 0.0,
        "split_source": str(split_dir),
        "train_ids": sorted(train_ids),
        "val_ids": sorted(val_ids),
        "test_ids": sorted(test_ids),
    }

    split_map_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(split_map_path, payload)

    return SplitResult(
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        test_ids=sorted(test_ids),
        split_map_path=split_map_path,
    )
