from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch



def save_checkpoint(path: Path, state: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)



def load_checkpoint(path: Path, map_location: str = "cpu") -> Dict[str, object]:
    return torch.load(path, map_location=map_location)
