#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import run_stage_preprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 01: preprocess and split (train/test=7:3).")
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    parser.add_argument(
        "--generate-demo-data",
        action="store_true",
        help="Generate small demo input file when running in lightweight mode.",
    )
    args = parser.parse_args()

    result = run_stage_preprocess(args.config, generate_demo_data=args.generate_demo_data)
    print(result.message)
    for f in result.files:
        print(f" - {f}")


if __name__ == "__main__":
    main()

