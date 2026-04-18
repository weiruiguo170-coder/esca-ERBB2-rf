from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_prediction_table  # noqa: E402
from src.utils import ensure_dir, load_config, project_root_from_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RF/LR/SVR metrics on test set.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    pred_table = pd.read_csv(resolve_path(root, paths["model_predictions"]))
    metrics = evaluate_prediction_table(pred_table)

    ensure_dir(resolve_path(root, paths["evaluation_dir"]))
    metrics.to_csv(resolve_path(root, paths["model_metrics"]), index=False)

    print("[06] Evaluation completed.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
