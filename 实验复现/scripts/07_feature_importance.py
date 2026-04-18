from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.importance import compute_feature_importance  # noqa: E402
from src.utils import ensure_dir, load_config, project_root_from_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute impurity and permutation importance.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    rf_model = joblib.load(resolve_path(root, paths["rf_model_file"]))
    X_test = pd.read_csv(resolve_path(root, paths["x_test"]), index_col="cell_line_id")
    y_test = pd.read_csv(resolve_path(root, paths["y_test"]), index_col="cell_line_id")[
        "joint_log2_ic50_mean"
    ]

    importance_table = compute_feature_importance(
        rf_model=rf_model,
        X_test=X_test,
        y_test=y_test,
        n_repeats=int(config["importance"]["permutation_repeats"]),
        random_state=int(config["importance"]["random_state"]),
    )

    ensure_dir(resolve_path(root, paths["importance_dir"]))
    importance_table.to_csv(resolve_path(root, paths["feature_importance_table"]), index=False)

    print("[07] Feature importance table created.")
    print(f"[07] Features ranked: {importance_table.shape[0]}")


if __name__ == "__main__":
    main()
