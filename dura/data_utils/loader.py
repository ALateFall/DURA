from __future__ import annotations


def shutdown_loader(loader) -> None:
    if loader is None:
        return

    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass

    try:
        loader._iterator = None
    except Exception:
        pass
