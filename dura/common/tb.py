from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_text(self, *args, **kwargs):
        return None

    def add_hparams(self, *args, **kwargs):
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def create_summary_writer(log_dir: Path) -> Tuple[object, Optional[str]]:
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))
        return writer, None
    except Exception as exc:  # noqa: BLE001
        return NullSummaryWriter(), str(exc)
