from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (  # noqa: E402
    fit_baselines,
    fit_random_forest,
    make_prediction_table,
    search_random_forest,
    search_svr,
)
from src.utils import ensure_dir, load_config, project_root_from_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF and baseline models.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    X_train = pd.read_csv(resolve_path(root, paths["x_train_bootstrap"]), index_col=0)
    y_train = pd.read_csv(resolve_path(root, paths["y_train_bootstrap"]), index_col=0)[
        "joint_log2_ic50_mean"
    ]
    X_test = pd.read_csv(resolve_path(root, paths["x_test"]), index_col="cell_line_id")
    y_test = pd.read_csv(resolve_path(root, paths["y_test"]), index_col="cell_line_id")[
        "joint_log2_ic50_mean"
    ]

    random_state = int(config["project"]["random_state"])

    rf_best_params, rf_search_table = search_random_forest(
        X_train=X_train,
        y_train=y_train,
        modeling_cfg=config["modeling"],
        random_state=random_state,
    )
    rf_model = fit_random_forest(
        X_train=X_train,
        y_train=y_train,
        params=rf_best_params,
        random_state=random_state,
    )

    svr_best_params, svr_search_table = search_svr(
        X_train=X_train,
        y_train=y_train,
        modeling_cfg=config["modeling"],
        random_state=random_state,
    )
    baselines = fit_baselines(X_train=X_train, y_train=y_train, svr_params=svr_best_params)

    pred_table = make_prediction_table(
        y_true=y_test,
        rf_model=rf_model,
        baseline_models=baselines,
        X_test=X_test,
    )

    ensure_dir(resolve_path(root, paths["modeling_dir"]))
    rf_search_table.to_csv(resolve_path(root, paths["rf_search_table"]), index=False)
    svr_search_table.to_csv(resolve_path(root, paths["svr_search_table"]), index=False)
    pred_table.to_csv(resolve_path(root, paths["model_predictions"]), index=False)

    joblib.dump(rf_model, resolve_path(root, paths["rf_model_file"]))

    selected_params = {
        "rf_selected_by_cv_oob": rf_best_params,
        "rf_paper_best": config["modeling"]["rf"]["paper_best"],
        "svr_selected_by_cv_r2": svr_best_params,
    }
    with resolve_path(root, paths["selected_params_file"]).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(selected_params, fh, sort_keys=False)

    print("[05] Model training completed.")
    print(f"[05] RF best params by (CV R2 + OOB R2)/2: {rf_best_params}")
    print(f"[05] Paper-reported RF params: {config['modeling']['rf']['paper_best']}")
    print(f"[05] SVR best params by CV R2: {svr_best_params}")


if __name__ == "__main__":
    main()
