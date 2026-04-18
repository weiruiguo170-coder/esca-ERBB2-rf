from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import bootstrap_training_set  # noqa: E402
from src.utils import ensure_dir, load_config, project_root_from_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap resampling on training set only.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    X_train = pd.read_csv(resolve_path(root, paths["x_train"]), index_col="cell_line_id")
    y_train = pd.read_csv(resolve_path(root, paths["y_train"]), index_col="cell_line_id")[
        "joint_log2_ic50_mean"
    ]

    X_boot, y_boot, sampled_ids = bootstrap_training_set(
        X_train=X_train,
        y_train=y_train,
        n_samples=int(config["bootstrap"]["n_samples"]),
        random_state=int(config["project"]["random_state"]),
    )

    ensure_dir(resolve_path(root, paths["bootstrap_dir"]))
    X_boot.to_csv(resolve_path(root, paths["x_train_bootstrap"]), index=True, index_label="bootstrap_id")
    y_boot.rename("joint_log2_ic50_mean").to_frame().to_csv(
        resolve_path(root, paths["y_train_bootstrap"]), index=True, index_label="bootstrap_id"
    )
    sampled_ids.to_frame().to_csv(resolve_path(root, paths["bootstrap_indices"]), index=False)

    print("[04] Bootstrap sampling completed.")
    print(f"[04] Generated samples: {len(X_boot)}")


if __name__ == "__main__":
    main()
