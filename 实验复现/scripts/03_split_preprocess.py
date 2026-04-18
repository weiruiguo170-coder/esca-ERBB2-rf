from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import split_and_preprocess  # noqa: E402
from src.utils import dump_json, ensure_dir, load_config, project_root_from_config, resolve_path, write_lines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test split and preprocessing.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def _feature_to_gene(feature_name: str) -> str:
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)

    paths = config["paths"]
    feature_path = resolve_path(root, paths["feature_matrix"])
    target_path = resolve_path(root, paths["target_table"])

    X = pd.read_csv(feature_path, index_col="cell_line_id")
    y = pd.read_csv(target_path, index_col="cell_line_id")["joint_log2_ic50_mean"]

    result = split_and_preprocess(
        X=X,
        y=y,
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["project"]["random_state"]),
        variance_threshold=float(config["preprocess"]["variance_threshold"]),
        expression_prefix=config["preprocess"]["expression_prefix"],
    )

    preprocess_dir = resolve_path(root, paths["preprocess_dir"])
    ensure_dir(preprocess_dir)

    result["X_train"].to_csv(resolve_path(root, paths["x_train"]), index=True, index_label="cell_line_id")
    result["X_test"].to_csv(resolve_path(root, paths["x_test"]), index=True, index_label="cell_line_id")
    result["y_train"].rename("joint_log2_ic50_mean").to_frame().to_csv(
        resolve_path(root, paths["y_train"]), index=True, index_label="cell_line_id"
    )
    result["y_test"].rename("joint_log2_ic50_mean").to_frame().to_csv(
        resolve_path(root, paths["y_test"]), index=True, index_label="cell_line_id"
    )

    write_lines(resolve_path(root, paths["selected_features"]), result["selected_features"])

    selected_expr_genes = [_feature_to_gene(col) for col in result["selected_expression_cols"]]
    write_lines(resolve_path(root, paths["selected_expression_genes"]), selected_expr_genes)

    stats = {
        "random_state": int(config["project"]["random_state"]),
        "test_size": float(config["split"]["test_size"]),
        "variance_threshold": float(config["preprocess"]["variance_threshold"]),
        "input_samples": int(X.shape[0]),
        "input_features": int(X.shape[1]),
        "train_samples": int(result["X_train"].shape[0]),
        "test_samples": int(result["X_test"].shape[0]),
        "selected_features": int(len(result["selected_features"])),
        "selected_expression_features": int(len(result["selected_expression_cols"])),
    }
    dump_json(resolve_path(root, paths["preprocess_stats"]), stats)

    print("[03] Split and preprocessing completed.")
    print(
        f"[03] Train/Test: {stats['train_samples']}/{stats['test_samples']}, "
        f"selected expression features: {stats['selected_expression_features']}"
    )


if __name__ == "__main__":
    main()
