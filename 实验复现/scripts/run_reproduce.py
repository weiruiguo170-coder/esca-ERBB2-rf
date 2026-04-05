#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lightweight_repro_pipeline import (
    run_demo_pipeline,
    run_stage_baselines,
    run_stage_bootstrap,
    run_stage_feature_importance,
    run_stage_preprocess,
    run_stage_train_rf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified entry for lightweight reproducibility workflow.")
    parser.add_argument(
        "--mode",
        default="demo",
        choices=["demo", "preprocess", "bootstrap", "train_rf", "baselines", "importance"],
        help="Workflow stage to execute.",
    )
    parser.add_argument("--config", default="configs/config.demo.yaml", help="Path to config yaml.")
    parser.add_argument("--strategy", default=None, choices=["paper_best", "grid_search"], help="RF strategy override.")
    parser.add_argument("--full-importance", action="store_true", help="Use full permutation repeats in importance step.")
    parser.add_argument("--force-demo-data", action="store_true", help="Regenerate demo input data.")
    args = parser.parse_args()

    if args.mode == "demo":
        results = run_demo_pipeline(args.config, force_demo_data=args.force_demo_data)
        for stage in results:
            print(stage.message)
            for f in stage.files:
                print(f" - {f}")
        return

    if args.mode == "preprocess":
        out = run_stage_preprocess(args.config, generate_demo_data=False)
    elif args.mode == "bootstrap":
        out = run_stage_bootstrap(args.config)
    elif args.mode == "train_rf":
        out = run_stage_train_rf(args.config, strategy=args.strategy)
    elif args.mode == "baselines":
        out = run_stage_baselines(args.config)
    else:
        mode = "full" if args.full_importance else "demo"
        out = run_stage_feature_importance(args.config, mode=mode)

    print(out.message)
    for f in out.files:
        print(f" - {f}")


if __name__ == "__main__":
    main()

