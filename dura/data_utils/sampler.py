from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import torch
from torch.utils.data import WeightedRandomSampler

from dura.data_utils.cleaning import NewsRecord



def build_sampler(strategy: str, uids: List[str], records_by_uid: Dict[str, NewsRecord]) -> Optional[WeightedRandomSampler]:
    if strategy == "none":
        return None

    if not uids:
        return None

    if strategy == "class":
        keys = [str(records_by_uid[uid].label) for uid in uids]
    elif strategy == "domain":
        keys = [records_by_uid[uid].domain for uid in uids]
    elif strategy == "class_domain":
        keys = [f"{records_by_uid[uid].label}_{records_by_uid[uid].domain}" for uid in uids]
    else:
        raise ValueError(f"Unsupported sampler strategy: {strategy}")

    counts = Counter(keys)
    weights = torch.tensor([1.0 / counts[k] for k in keys], dtype=torch.double)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
