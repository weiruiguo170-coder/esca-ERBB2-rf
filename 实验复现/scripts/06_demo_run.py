#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import run_demo_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 06: one-command lightweight demo run.")
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    parser.add_argument(
        "--force-demo-data",
        action="store_true",
        help="Regenerate demo input CSV before running.",
    )
    args = parser.parse_args()

    results = run_demo_pipeline(args.config, force_demo_data=args.force_demo_data)
    for stage in results:
        print(stage.message)
        for f in stage.files:
            print(f" - {f}")


if __name__ == "__main__":
    main()

