from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    return sha256_text(payload)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
