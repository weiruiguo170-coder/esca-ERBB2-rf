#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_cell_line_name(name: str) -> str:
    text = str(name).strip().upper()
    for token in [" ", "-", "_", ".", "/", "(", ")"]:
        text = text.replace(token, "")
    return text


def audit_input_files(file_map: Dict[str, Path]) -> pd.DataFrame:
    rows = []
    for key, path in file_map.items():
        exists = path.exists()
        rows.append(
            {
                "file_key": key,
                "path": str(path),
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _find_col(columns: Iterable[str], keywords: Iterable[str]) -> str | None:
    lower = [c.lower() for c in columns]
    for kw in keywords:
        for idx, col in enumerate(lower):
            if kw in col:
                return list(columns)[idx]
    return None


def read_gdsc(gdsc_path: Path) -> pd.DataFrame:
    return pd.read_excel(gdsc_path)


def read_screened_compounds(comp_path: Path) -> pd.DataFrame:
    return pd.read_csv(comp_path)


def read_cell_annotations(anno_path: Path) -> pd.DataFrame:
    return pd.read_csv(anno_path, sep="\t", dtype=str)


def _keyword_mask(series: pd.Series, keywords: List[str]) -> pd.Series:
    text = series.fillna("").str.lower()
    mask = pd.Series(False, index=series.index)
    for kw in keywords:
        mask = mask | text.str.contains(kw.lower(), regex=False)
    return mask


def identify_her2_drugs(
    screened_compounds: pd.DataFrame, keywords: List[str]
) -> pd.DataFrame:
    work = screened_compounds.copy()
    cols = list(work.columns)
    target_col = _find_col(cols, ["target", "putative_target"])
    pathway_col = _find_col(cols, ["pathway"])
    name_col = _find_col(cols, ["drug_name", "name", "compound"])
    if not name_col:
        raise ValueError("screened_compounds 未检测到药物名称列。")

    name_mask = _keyword_mask(work[name_col], keywords)
    target_mask = _keyword_mask(work[target_col], keywords) if target_col else False
    pathway_mask = _keyword_mask(work[pathway_col], keywords) if pathway_col else False
    mask = name_mask | target_mask | pathway_mask
    out = work.loc[mask].copy()
    out["selection_basis"] = np.select(
        [name_mask.loc[mask], target_mask.loc[mask], pathway_mask.loc[mask]],
        ["name", "target", "pathway"],
        default="mixed",
    )
    return out


def filter_esophageal_cell_lines(
    annotations: pd.DataFrame, keywords: List[str]
) -> pd.DataFrame:
    work = annotations.copy()
    cols = list(work.columns)
    tissue_col = _find_col(cols, ["primary", "lineage", "disease", "site", "cancer", "hist"])
    name_col = _find_col(cols, ["ccle_name", "cell_line", "name"])
    if not name_col:
        raise ValueError("annotations 未检测到细胞系名称列。")
    if tissue_col:
        mask = _keyword_mask(work[tissue_col], keywords)
    else:
        mask = _keyword_mask(work[name_col], keywords)
    out = work.loc[mask].copy()
    out["normalized_cell_line"] = out[name_col].map(normalize_cell_line_name)
    return out


def build_matching_report(
    gdsc_df: pd.DataFrame, anno_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gdsc = gdsc_df.copy()
    ann = anno_df.copy()
    gdsc_name_col = _find_col(gdsc.columns, ["cell_line_name", "cell line", "cell_line"])
    ann_name_col = _find_col(ann.columns, ["ccle_name", "cell_line", "name"])
    if not gdsc_name_col or not ann_name_col:
        raise ValueError("缺少细胞系名称列，无法构建匹配报告。")

    gdsc["normalized_name"] = gdsc[gdsc_name_col].map(normalize_cell_line_name)
    ann["normalized_name"] = ann[ann_name_col].map(normalize_cell_line_name)
    ann_map = (
        ann.drop_duplicates("normalized_name")
        .set_index("normalized_name")[ann_name_col]
        .to_dict()
    )

    gdsc["matched_ccle_name"] = gdsc["normalized_name"].map(ann_map)
    gdsc["match_status"] = np.where(gdsc["matched_ccle_name"].notna(), "matched", "unmatched")
    report_cols = [gdsc_name_col, "normalized_name", "matched_ccle_name", "match_status"]
    report = gdsc[report_cols].drop_duplicates().rename(columns={gdsc_name_col: "gdsc_cell_line"})
    matched = gdsc[gdsc["match_status"] == "matched"].copy()
    return report, matched


def _gene_symbol(raw_gene: str) -> str:
    gene = str(raw_gene).strip()
    if "|" in gene:
        gene = gene.split("|")[0]
    return gene.upper()


def extract_focus_expression_from_gct(
    gct_path: Path, focus_genes: List[str]
) -> pd.DataFrame:
    targets = {g.upper() for g in focus_genes}
    with gzip.open(gct_path, "rt", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        _ = f.readline()
        header = f.readline().rstrip("\n").split("\t")
        sample_cols = header[2:]
        rows: List[Dict] = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            gene = _gene_symbol(parts[0])
            if gene not in targets:
                continue
            expr_vals = pd.to_numeric(pd.Series(parts[2:]), errors="coerce").to_numpy(dtype=float)
            rows.append({"gene": gene, **{sample_cols[i]: expr_vals[i] for i in range(len(sample_cols))}})
    if not rows:
        return pd.DataFrame(columns=["sample_id"] + [f"EXP_{g}" for g in focus_genes])

    wide = pd.DataFrame(rows).drop_duplicates("gene").set_index("gene").T
    wide.index.name = "sample_id"
    wide = wide.reset_index()
    wide.columns = ["sample_id"] + [f"EXP_{c}" for c in wide.columns[1:]]
    return wide


def extract_erbb2_cnv(cnv_path: Path) -> pd.DataFrame:
    df = pd.read_excel(cnv_path)
    cols = list(df.columns)
    sample_col = _find_col(cols, ["ccle_name", "cell_line", "sample", "name"])
    erbb2_col = _find_col(cols, ["erbb2"])
    if not sample_col or not erbb2_col:
        return pd.DataFrame(columns=["sample_id", "ERBB2_CNV"])
    out = df[[sample_col, erbb2_col]].copy()
    out.columns = ["sample_id", "ERBB2_CNV"]
    out["ERBB2_CNV"] = pd.to_numeric(out["ERBB2_CNV"], errors="coerce")
    return out


def build_merged_dataset(
    cfg: Dict, output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = cfg["paths"]
    raw = Path(paths["raw_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = {
        "ccle_expression_gct": raw / cfg["raw_files"]["ccle_expression_gct"],
        "ccle_absolute_xlsx": raw / cfg["raw_files"]["ccle_absolute_xlsx"],
        "cell_annotations_txt": raw / cfg["raw_files"]["cell_annotations_txt"],
        "gdsc2_dose_response_xlsx": raw / cfg["raw_files"]["gdsc2_dose_response_xlsx"],
        "screened_compounds_csv": raw / cfg["raw_files"]["screened_compounds_csv"],
    }
    audit_df = audit_input_files(raw_files)
    audit_df.to_csv(output_dir / "input_audit.csv", index=False, encoding="utf-8-sig")

    missing = audit_df.loc[~audit_df["exists"], "file_key"].tolist()
    if missing:
        raise FileNotFoundError(f"缺少原始数据文件: {missing}")

    gdsc = read_gdsc(raw_files["gdsc2_dose_response_xlsx"])
    compounds = read_screened_compounds(raw_files["screened_compounds_csv"])
    anno = read_cell_annotations(raw_files["cell_annotations_txt"])

    her2_drugs = identify_her2_drugs(compounds, cfg["drug_filters"]["keywords"])
    her2_drugs.to_csv(output_dir / "candidate_her2_drugs.csv", index=False, encoding="utf-8-sig")

    gdsc_drug_col = _find_col(gdsc.columns, ["drug_name", "drug"])
    ln_ic50_col = _find_col(gdsc.columns, ["ln_ic50"])
    if not gdsc_drug_col or not ln_ic50_col:
        raise ValueError("GDSC2 文件缺少药物名或 LN_IC50 列。")
    gdsc = gdsc.copy()
    gdsc[ln_ic50_col] = pd.to_numeric(gdsc[ln_ic50_col], errors="coerce")
    gdsc = gdsc[gdsc[ln_ic50_col].notna()].copy()

    hname_col = _find_col(her2_drugs.columns, ["drug_name", "name", "compound"])
    selected_drugs = set(her2_drugs[hname_col].dropna().astype(str))
    gdsc = gdsc[gdsc[gdsc_drug_col].astype(str).isin(selected_drugs)].copy()

    esca = filter_esophageal_cell_lines(anno, cfg["cell_line_filters"]["keywords"])
    esca.to_csv(output_dir / "esophageal_like_cell_lines.csv", index=False, encoding="utf-8-sig")

    report, gdsc_matched = build_matching_report(gdsc, esca)
    report.to_csv(output_dir / "matching_report.csv", index=False, encoding="utf-8-sig")

    expr = extract_focus_expression_from_gct(
        raw_files["ccle_expression_gct"], cfg["feature_engineering"]["focus_genes"]
    )
    expr["normalized_name"] = expr["sample_id"].map(normalize_cell_line_name)
    cnv = extract_erbb2_cnv(raw_files["ccle_absolute_xlsx"])
    cnv["normalized_name"] = cnv["sample_id"].map(normalize_cell_line_name)

    gdsc_cell_col = _find_col(gdsc_matched.columns, ["cell_line_name", "cell line", "cell_line"])
    gdsc_matched["normalized_name"] = gdsc_matched[gdsc_cell_col].map(normalize_cell_line_name)

    merged = gdsc_matched.merge(expr, on="normalized_name", how="left").merge(
        cnv[["normalized_name", "ERBB2_CNV"]], on="normalized_name", how="left"
    )

    merged = pd.get_dummies(merged, columns=[gdsc_drug_col], prefix="DRUG", dtype=float)
    merged = merged.rename(columns={ln_ic50_col: "LN_IC50"})
    merged.to_csv(output_dir / "merged_dataset.csv", index=False, encoding="utf-8-sig")

    meta = {
        "n_rows": int(len(merged)),
        "n_cols": int(merged.shape[1]),
        "n_candidate_drugs": int(len(selected_drugs)),
    }
    with (output_dir / "merged_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return merged, report, her2_drugs


def load_or_build_dataset(cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inter_dir = Path(cfg["paths"]["intermediate_dir"])
    inter_dir.mkdir(parents=True, exist_ok=True)
    merged_path = inter_dir / "merged_dataset.csv"
    report_path = inter_dir / "matching_report.csv"

    if cfg["pipeline"].get("prefer_cached_intermediate", True) and merged_path.exists():
        merged = pd.read_csv(merged_path)
        report = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
        return merged, report

    merged, report, _ = build_merged_dataset(cfg, inter_dir)
    return merged, report
