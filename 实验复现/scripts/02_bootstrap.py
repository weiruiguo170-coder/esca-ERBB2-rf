#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import run_stage_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 02: bootstrap on training set only.")
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    args = parser.parse_args()

    result = run_stage_bootstrap(args.config)
    print(result.message)
    for f in result.files:
        print(f" - {f}")


if __name__ == "__main__":
    main()

