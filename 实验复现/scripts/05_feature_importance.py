#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import run_stage_feature_importance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 05: impurity + permutation importance, then Spearman clustering."
    )
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    parser.add_argument("--mode", default="demo", choices=["demo", "full"], help="Controls permutation repeats.")
    args = parser.parse_args()

    result = run_stage_feature_importance(args.config, mode=args.mode)
    print(result.message)
    for f in result.files:
        print(f" - {f}")


if __name__ == "__main__":
    main()

