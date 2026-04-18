from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_feature_matrix, compute_joint_target  # noqa: E402
from src.utils import load_config, project_root_from_config, read_lines, resolve_path, ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build joint endpoint and feature matrix.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)

    paths = config["paths"]
    expression = pd.read_csv(resolve_path(root, paths["prepared_expression"]), index_col="gene_symbol")
    cnv = pd.read_csv(resolve_path(root, paths["prepared_cnv"]), index_col="gene_symbol")
    drug = pd.read_csv(resolve_path(root, paths["prepared_drug_response"]))
    pathway_genes = read_lines(resolve_path(root, paths["prepared_pathway_genes"]))

    target = compute_joint_target(drug)

    features, pathway_genes_present = build_feature_matrix(
        expression=expression,
        cnv=cnv,
        pathway_genes=pathway_genes,
        expression_prefix=config["preprocess"]["expression_prefix"],
        cnv_prefix=config["preprocess"]["cnv_prefix"],
    )

    aligned_ids = features.index.intersection(target.index)
    if len(aligned_ids) < 3:
        raise ValueError("Too few samples remain after target/feature alignment.")

    features = features.loc[aligned_ids]
    target = target.loc[aligned_ids]

    ensure_dir(resolve_path(root, paths["feature_dir"]))
    feature_out = resolve_path(root, paths["feature_matrix"])
    target_out = resolve_path(root, paths["target_table"])

    features.index.name = "cell_line_id"
    features.to_csv(feature_out, index=True)
    target.rename("joint_log2_ic50_mean").to_frame().to_csv(target_out, index=True, index_label="cell_line_id")

    print("[02] Feature matrix and target table created.")
    print(
        f"[02] Samples: {len(aligned_ids)}, total features: {features.shape[1]}, "
        f"CNV pathway genes used: {len(pathway_genes_present)}"
    )


if __name__ == "__main__":
    main()
