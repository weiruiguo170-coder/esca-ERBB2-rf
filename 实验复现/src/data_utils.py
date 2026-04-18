from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_DRUG_COLUMNS = [
    "cell_line_id",
    "source",
    "tissue_group",
    "erbb2_axis_relevant",
    "trastuzumab_ic50",
    "lapatinib_ic50",
]


def _load_gene_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}. Place the public-data matrix in data/input/."
        )

    matrix = pd.read_csv(path)
    if "gene_symbol" not in matrix.columns:
        raise ValueError(f"{path.name} must contain a 'gene_symbol' column.")

    matrix["gene_symbol"] = matrix["gene_symbol"].astype(str).str.strip()
    matrix = matrix.dropna(subset=["gene_symbol"])

    if matrix["gene_symbol"].duplicated().any():
        numeric_cols = [c for c in matrix.columns if c != "gene_symbol"]
        matrix[numeric_cols] = matrix[numeric_cols].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.groupby("gene_symbol", as_index=False)[numeric_cols].mean()

    matrix = matrix.set_index("gene_symbol")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.shape[1] == 0:
        raise ValueError(f"{path.name} has no cell-line columns.")
    return matrix


def load_expression_matrix(path: Path) -> pd.DataFrame:
    return _load_gene_matrix(path)


def load_cnv_matrix(path: Path) -> pd.DataFrame:
    return _load_gene_matrix(path)


def load_drug_response(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}. Place the public-data table in data/input/."
        )

    table = pd.read_csv(path)
    missing = [col for col in REQUIRED_DRUG_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}. "
            "See data/schema/drug_response_template.csv"
        )

    table = table[REQUIRED_DRUG_COLUMNS].copy()
    table["cell_line_id"] = table["cell_line_id"].astype(str).str.strip()
    table["source"] = table["source"].astype(str).str.upper().str.strip()
    table["tissue_group"] = table["tissue_group"].astype(str).str.strip().str.lower()
    table["erbb2_axis_relevant"] = pd.to_numeric(
        table["erbb2_axis_relevant"], errors="coerce"
    )
    table["trastuzumab_ic50"] = pd.to_numeric(table["trastuzumab_ic50"], errors="coerce")
    table["lapatinib_ic50"] = pd.to_numeric(table["lapatinib_ic50"], errors="coerce")

    table = table.dropna(subset=["cell_line_id"])
    table = table.drop_duplicates(subset=["cell_line_id"], keep="first")
    return table


def load_pathway_genes(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}. Provide ERBB2-pathway genes in data/input/."
        )

    genes: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            token = line.strip()
            if token:
                genes.append(token)

    if not genes:
        raise ValueError(f"{path.name} is empty.")
    return sorted(set(genes))


def filter_and_align_data(
    expression: pd.DataFrame,
    cnv: pd.DataFrame,
    drug: pd.DataFrame,
    tissue_group_value: str,
    erbb2_axis_flag: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    filtered = drug.copy()
    filtered = filtered[filtered["source"].isin(["GDSC", "CCLE"])]
    filtered = filtered[filtered["tissue_group"] == tissue_group_value.lower()]
    filtered = filtered[filtered["erbb2_axis_relevant"] == erbb2_axis_flag]

    if filtered.empty:
        raise ValueError("No rows remain after filtering by source/tissue/ERBB2-axis flag.")

    cell_lines = sorted(
        set(filtered["cell_line_id"]).intersection(expression.columns).intersection(cnv.columns)
    )
    if len(cell_lines) < 3:
        raise ValueError(
            "Too few aligned cell lines after filtering. "
            "Check IDs across expression/cnv/drug response files."
        )

    expression_aligned = expression.loc[:, cell_lines]
    cnv_aligned = cnv.loc[:, cell_lines]
    filtered_aligned = filtered[filtered["cell_line_id"].isin(cell_lines)].copy()
    filtered_aligned = filtered_aligned.set_index("cell_line_id").loc[cell_lines].reset_index()

    return expression_aligned, cnv_aligned, filtered_aligned
