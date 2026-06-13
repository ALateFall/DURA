#!/usr/bin/env python
from __future__ import annotations

from _bootstrap import bootstrap

bootstrap()

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run a DURA training and evaluation pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to a DURA YAML configuration file.")
    parser.add_argument(
        "--override",
        type=str,
        nargs="+",
        action="append",
        default=[],
        help="Override config values with dotted key=value entries. Can be used multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from dura.config import load_config
    from dura.pipeline import run_pipeline

    overrides = [item for group in args.override for item in group]
    cfg = load_config(args.config, overrides=overrides)
    summary = run_pipeline(cfg)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
