from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_utils import (  # noqa: E402
    filter_and_align_data,
    load_cnv_matrix,
    load_drug_response,
    load_expression_matrix,
    load_pathway_genes,
)
from src.utils import ensure_dir, resolve_path, write_lines, load_config, project_root_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and align public GDSC/CCLE data.")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)

    paths = config["paths"]
    expression_path = resolve_path(root, paths["expression_input"])
    cnv_path = resolve_path(root, paths["cnv_input"])
    drug_path = resolve_path(root, paths["drug_response_input"])
    pathway_path = resolve_path(root, paths["pathway_genes_input"])

    prepared_dir = resolve_path(root, paths["prepared_dir"])
    ensure_dir(prepared_dir)

    expression = load_expression_matrix(expression_path)
    cnv = load_cnv_matrix(cnv_path)
    drug = load_drug_response(drug_path)
    pathway_genes = load_pathway_genes(pathway_path)

    expression_aligned, cnv_aligned, drug_aligned = filter_and_align_data(
        expression=expression,
        cnv=cnv,
        drug=drug,
        tissue_group_value=config["data"]["tissue_group_value"],
        erbb2_axis_flag=int(config["data"]["erbb2_axis_flag_value"]),
    )

    expression_out = resolve_path(root, paths["prepared_expression"])
    cnv_out = resolve_path(root, paths["prepared_cnv"])
    drug_out = resolve_path(root, paths["prepared_drug_response"])
    cell_lines_out = resolve_path(root, paths["prepared_cell_lines"])
    pathway_out = resolve_path(root, paths["prepared_pathway_genes"])

    expression_aligned.to_csv(expression_out, index=True, index_label="gene_symbol")
    cnv_aligned.to_csv(cnv_out, index=True, index_label="gene_symbol")
    drug_aligned.to_csv(drug_out, index=False)

    pd.DataFrame({"cell_line_id": expression_aligned.columns}).to_csv(
        cell_lines_out, index=False
    )
    write_lines(pathway_out, pathway_genes)

    print("[01] Prepared data written to", prepared_dir)
    print(
        f"[01] Aligned cell lines: {len(expression_aligned.columns)}, "
        f"expression genes: {expression_aligned.shape[0]}, cnv genes: {cnv_aligned.shape[0]}"
    )


if __name__ == "__main__":
    main()
