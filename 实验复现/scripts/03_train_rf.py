#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import run_stage_train_rf


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 03: train/select RandomForest using CV R2 + OOB R2.")
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["paper_best", "grid_search"],
        help="Override model selection strategy.",
    )
    args = parser.parse_args()

    result = run_stage_train_rf(args.config, strategy=args.strategy)
    print(result.message)
    for f in result.files:
        print(f" - {f}")


if __name__ == "__main__":
    main()

