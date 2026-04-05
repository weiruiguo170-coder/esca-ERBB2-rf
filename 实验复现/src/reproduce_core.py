#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATH_GDSC = os.path.join(BASE_DIR, "GDSC2_fitted_dose_response_27Oct23 .xlsx")
PATH_GCT = os.path.join(BASE_DIR, "CCLE_RNAseq_genes_rpkm_20180929.gct.gz")
PATH_ABSOLUTE = os.path.join(BASE_DIR, "CCLE_ABSOLUTE_combined_20181227.xlsx")
PATH_ANNOT = os.path.join(BASE_DIR, "Cell_lines_annotations_20181226.txt")
PATH_CANDIDATE_DRUGS = os.path.join(BASE_DIR, "results", "candidate_drugs.csv")
PATH_ESO_LINES = os.path.join(BASE_DIR, "results", "esophageal_cell_lines.csv")

DIR_INTERMEDIATE = os.path.join(BASE_DIR, "intermediate")
DIR_RESULTS = os.path.join(BASE_DIR, "results")
DIR_FIGURES = os.path.join(BASE_DIR, "figures")
DIR_LOGS = os.path.join(BASE_DIR, "logs")

FOCUS_GENES = [
    "ERBB2",
    "GRB7",
    "ERBB3",
    "PIK3CA",
    "AKT1",
    "MAPK1",
    "PTK6",
    "CCND1",
    "SHC1",
]
ERBB2_CHR = 17
ERBB2_START = 39723979
ERBB2_END = 39750390


@dataclass
class RunContext:
    merged_rows: int
    merged_feature_cols: int
    bootstrap_rows: int
    selected_feature_cols: int
    missing_focus_genes: List[str]
    ln_ic50_note: str


def ensure_dirs() -> None:
    for d in [DIR_INTERMEDIATE, DIR_RESULTS, DIR_FIGURES, DIR_LOGS]:
        os.makedirs(d, exist_ok=True)


def norm_name(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).upper()
    return "".join(ch for ch in text if ch.isalnum())


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gdsc = pd.read_excel(PATH_GDSC, sheet_name=0)
    annot = pd.read_csv(PATH_ANNOT, sep="\t")
    candidate = pd.read_csv(PATH_CANDIDATE_DRUGS)
    eso = pd.read_csv(PATH_ESO_LINES)
    return gdsc, annot, candidate, eso


def build_matching(
    gdsc: pd.DataFrame, annot: pd.DataFrame, candidate: pd.DataFrame, eso: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    candidate_names = set(candidate["DRUG_NAME"].dropna().astype(str).tolist())
    gdsc2 = gdsc[gdsc["DRUG_NAME"].isin(candidate_names)].copy()
    gdsc2["normalized_name"] = gdsc2["CELL_LINE_NAME"].map(norm_name)

    annot = annot.copy()
    annot["normalized_name"] = annot["Name"].map(norm_name)
    eso_ids = set(eso["CCLE_ID"].dropna().astype(str).tolist())

    grouped = annot.groupby("normalized_name", dropna=False)
    unique_map: Dict[str, pd.Series] = {}
    ambiguous_map: Dict[str, pd.DataFrame] = {}
    for k, g in grouped:
        if not k:
            continue
        if len(g) == 1:
            unique_map[k] = g.iloc[0]
        else:
            ambiguous_map[k] = g

    records = []
    for row in gdsc2.itertuples(index=False):
        n = row.normalized_name
        match_status = "匹配失败"
        ccle_id = ""
        depmap_id = ""
        ccle_name = ""
        match_rule = "无"
        if n in unique_map:
            item = unique_map[n]
            match_status = "匹配成功"
            ccle_id = str(item["CCLE_ID"])
            depmap_id = str(item["depMapID"])
            ccle_name = str(item["Name"])
            match_rule = "标准化名称精确匹配"
        elif n in ambiguous_map:
            g = ambiguous_map[n]
            match_status = "待人工确认"
            ccle_id = ";".join(g["CCLE_ID"].astype(str).tolist())
            depmap_id = ";".join(g["depMapID"].astype(str).tolist())
            ccle_name = ";".join(g["Name"].astype(str).tolist())
            match_rule = "标准化名称命中多个候选"

        is_eso = "是" if (ccle_id in eso_ids) else "否"
        records.append(
            {
                "DRUG_NAME": row.DRUG_NAME,
                "DRUG_ID": row.DRUG_ID,
                "CELL_LINE_NAME": row.CELL_LINE_NAME,
                "normalized_name": n,
                "match_status": match_status,
                "matched_CCLE_ID": ccle_id,
                "matched_DepMapID": depmap_id,
                "matched_CCLE_name": ccle_name,
                "match_rule": match_rule,
                "is_esophageal_or_gej": is_eso,
            }
        )

    matching = pd.DataFrame(records).drop_duplicates(
        subset=["DRUG_NAME", "CELL_LINE_NAME", "match_status", "matched_CCLE_ID"]
    )
    matching.to_csv(os.path.join(DIR_RESULTS, "matching_report.csv"), index=False, encoding="utf-8-sig")

    target_pairs = matching[
        (matching["match_status"] == "匹配成功") & (matching["is_esophageal_or_gej"] == "是")
    ][["DRUG_NAME", "CELL_LINE_NAME", "matched_CCLE_ID", "matched_DepMapID"]].drop_duplicates()

    gdsc_target = gdsc2.merge(
        target_pairs,
        how="inner",
        on=["DRUG_NAME", "CELL_LINE_NAME"],
    )
    gdsc_target = gdsc_target.rename(
        columns={"matched_CCLE_ID": "CCLE_ID", "matched_DepMapID": "depMapID"}
    )
    return matching, gdsc_target


def read_gct_selected_samples(sample_ids: List[str]) -> pd.DataFrame:
    with gzip.open(PATH_GCT, "rt", encoding="utf-8") as f:
        _ = f.readline().rstrip("\n")
        _ = f.readline().rstrip("\n")
        header = f.readline().rstrip("\n").split("\t")

    available_samples = [c for c in sample_ids if c in header]
    if len(available_samples) == 0:
        raise RuntimeError("GCT 中没有找到与目标队列匹配的 CCLE_ID 样本列。")

    usecols = ["Name", "Description"] + available_samples
    expr_raw = pd.read_csv(
        PATH_GCT,
        sep="\t",
        compression="gzip",
        skiprows=2,
        usecols=usecols,
        low_memory=False,
    )

    expr_raw["gene_symbol"] = expr_raw["Description"].fillna("").astype(str).str.strip()
    empty_symbol = expr_raw["gene_symbol"] == ""
    expr_raw.loc[empty_symbol, "gene_symbol"] = expr_raw.loc[empty_symbol, "Name"].astype(str)

    expr = expr_raw.groupby("gene_symbol", as_index=True)[available_samples].mean()
    expr_t = expr.T
    expr_t.index.name = "CCLE_ID"
    expr_t.columns = [f"EXP_{c}" for c in expr_t.columns]
    expr_t = expr_t.reset_index()
    return expr_t


def extract_erbb2_cnv() -> pd.DataFrame:
    seg = pd.read_excel(
        PATH_ABSOLUTE,
        sheet_name="ABSOLUTE_combined.segtab",
        usecols=[
            "sample",
            "Chromosome",
            "Start",
            "End",
            "Modal_Total_CN",
            "LOH",
            "Homozygous_deletion",
            "depMapID",
        ],
    )
    seg = seg[seg["Chromosome"] == ERBB2_CHR].copy()
    seg = seg[(seg["Start"] <= ERBB2_END) & (seg["End"] >= ERBB2_START)].copy()
    if seg.empty:
        raise RuntimeError("未在 ABSOLUTE segtab 中找到 ERBB2 基因位点重叠片段。")

    seg["overlap_len"] = np.minimum(seg["End"], ERBB2_END) - np.maximum(seg["Start"], ERBB2_START) + 1
    seg = seg.sort_values(["sample", "overlap_len"], ascending=[True, False])
    best = seg.groupby("sample", as_index=False).first()
    best = best.rename(
        columns={
            "sample": "CCLE_ID",
            "Modal_Total_CN": "ERBB2_CNV_MODAL_TOTAL_CN",
            "LOH": "ERBB2_CNV_LOH",
            "Homozygous_deletion": "ERBB2_CNV_HOMDEL",
        }
    )
    return best[
        ["CCLE_ID", "depMapID", "ERBB2_CNV_MODAL_TOTAL_CN", "ERBB2_CNV_LOH", "ERBB2_CNV_HOMDEL"]
    ].drop_duplicates()


def build_merged_dataset(gdsc_target: pd.DataFrame, expr_t: pd.DataFrame, cnv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    merged = gdsc_target.merge(expr_t, how="left", on="CCLE_ID")
    merged = merged.merge(cnv, how="left", on=["CCLE_ID", "depMapID"])

    merged["LN_IC50"] = pd.to_numeric(merged["LN_IC50"], errors="coerce")
    merged = merged.dropna(subset=["LN_IC50"]).copy()

    drug_dummies = pd.get_dummies(merged["DRUG_NAME"], prefix="DRUG")
    merged = pd.concat([merged.reset_index(drop=True), drug_dummies.reset_index(drop=True)], axis=1)

    feature_cols = [c for c in merged.columns if c.startswith("EXP_")]
    feature_cols += ["ERBB2_CNV_MODAL_TOTAL_CN", "ERBB2_CNV_LOH", "ERBB2_CNV_HOMDEL"]
    feature_cols += [
        c for c in merged.columns if c.startswith("DRUG_") and c not in {"DRUG_ID", "DRUG_NAME"}
    ]

    missing_focus = []
    for gene in FOCUS_GENES:
        col = f"EXP_{gene}"
        if col not in merged.columns:
            missing_focus.append(gene)
            merged[col] = np.nan
            feature_cols.append(col)

    feature_cols = sorted(set(feature_cols))
    cols_out = [
        "DATASET",
        "CELL_LINE_NAME",
        "CCLE_ID",
        "depMapID",
        "CANCER_TYPE",
        "DRUG_ID",
        "DRUG_NAME",
        "PUTATIVE_TARGET",
        "PATHWAY_NAME",
        "LN_IC50",
        "AUC",
        "RMSE",
        "Z_SCORE",
    ] + feature_cols
    cols_out = [c for c in cols_out if c in merged.columns]

    merged_out = merged[cols_out].copy()
    merged_out.to_csv(os.path.join(DIR_INTERMEDIATE, "merged_dataset.csv"), index=False, encoding="utf-8-sig")
    return merged_out, missing_focus


def preprocess_and_bootstrap(merged: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], RunContext]:
    metadata_cols = [
        "DATASET",
        "CELL_LINE_NAME",
        "CCLE_ID",
        "depMapID",
        "CANCER_TYPE",
        "DRUG_ID",
        "DRUG_NAME",
        "PUTATIVE_TARGET",
        "PATHWAY_NAME",
        "AUC",
        "RMSE",
        "Z_SCORE",
    ]
    feature_cols = [c for c in merged.columns if c not in metadata_cols + ["LN_IC50"]]

    X_raw = merged[feature_cols].copy()
    y = merged["LN_IC50"].astype(float).copy()

    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()
    vt = VarianceThreshold(threshold=0.01)

    X_imp = imputer.fit_transform(X_raw)
    X_scaled = scaler.fit_transform(X_imp)
    X_sel = vt.fit_transform(X_scaled)
    selected_cols = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]

    preprocessed = pd.DataFrame(X_sel, columns=selected_cols, index=merged.index)
    preprocessed["LN_IC50"] = y.values
    preprocessed["CCLE_ID"] = merged["CCLE_ID"].values
    preprocessed["DRUG_NAME"] = merged["DRUG_NAME"].values
    preprocessed["CELL_LINE_NAME"] = merged["CELL_LINE_NAME"].values

    rng = np.random.default_rng(42)
    real_n = len(preprocessed)
    if real_n == 0:
        raise RuntimeError("整合后样本数为 0，无法进行 Bootstrap 和模型训练。")
    boot_idx = rng.choice(np.arange(real_n), size=480, replace=True)
    boot = preprocessed.iloc[boot_idx].copy()
    boot.reset_index(drop=True, inplace=True)
    boot["bootstrap_source_index"] = boot_idx

    boot.to_csv(os.path.join(DIR_INTERMEDIATE, "bootstrap_dataset.csv"), index=False, encoding="utf-8-sig")

    comp_rows = []
    for var in ["LN_IC50"]:
        real_vals = preprocessed[var].astype(float)
        boot_vals = boot[var].astype(float)
        comp_rows.append(
            {
                "variable": var,
                "real_mean": float(real_vals.mean()),
                "real_std": float(real_vals.std(ddof=1)),
                "boot_mean": float(boot_vals.mean()),
                "boot_std": float(boot_vals.std(ddof=1)),
                "real_min": float(real_vals.min()),
                "real_max": float(real_vals.max()),
                "boot_min": float(boot_vals.min()),
                "boot_max": float(boot_vals.max()),
            }
        )
    pd.DataFrame(comp_rows).to_csv(
        os.path.join(DIR_RESULTS, "bootstrap_distribution_summary.csv"), index=False, encoding="utf-8-sig"
    )

    missing_focus = [g for g in FOCUS_GENES if f"EXP_{g}" not in merged.columns]
    ctx = RunContext(
        merged_rows=len(merged),
        merged_feature_cols=len(feature_cols),
        bootstrap_rows=len(boot),
        selected_feature_cols=len(selected_cols),
        missing_focus_genes=missing_focus,
        ln_ic50_note="GDSC2 字段名为 LN_IC50，按自然对数尺度处理；本次不额外转换为 log2(IC50)。",
    )
    return preprocessed, boot, selected_cols, ctx


def train_and_evaluate(
    boot: pd.DataFrame, selected_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, RandomForestRegressor]:
    X = boot[selected_cols].to_numpy(dtype=float)
    y = boot["LN_IC50"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Avoid nested parallelism explosion: parallelize CV level, keep each RF fit single-threaded.
    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_grid = {
        "n_estimators": [200, 300, 400, 500, 600, 700, 800],
        "max_depth": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "min_samples_split": [2, 4, 6, 8, 10],
    }
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    rf_cv_rmse = float(-rf_search.best_score_)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    lin_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_lin)))
    lin_r2 = float(r2_score(y_test, y_pred_lin))

    svr = SVR(kernel="rbf")
    svr_grid = {"C": [0.1, 1, 10, 100], "gamma": [1e-4, 1e-3, 1e-2, 1e-1]}
    svr_search = GridSearchCV(
        svr,
        svr_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    svr_search.fit(X_train, y_train)
    best_svr = svr_search.best_estimator_
    y_pred_svr = best_svr.predict(X_test)
    svr_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_svr)))
    svr_r2 = float(r2_score(y_test, y_pred_svr))

    metrics = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "test_rmse": rf_rmse,
                "test_r2": rf_r2,
                "cv_best_rmse": rf_cv_rmse,
                "best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
            }
        ]
    )

    model_compare = pd.DataFrame(
        [
            {
                "model": "RandomForestRegressor",
                "test_rmse": rf_rmse,
                "test_r2": rf_r2,
                "cv_best_rmse": rf_cv_rmse,
                "cv_best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
            },
            {
                "model": "LinearRegression",
                "test_rmse": lin_rmse,
                "test_r2": lin_r2,
                "cv_best_rmse": np.nan,
                "cv_best_params": "",
            },
            {
                "model": "SVR_RBF",
                "test_rmse": svr_rmse,
                "test_r2": svr_r2,
                "cv_best_rmse": float(-svr_search.best_score_),
                "cv_best_params": json.dumps(svr_search.best_params_, ensure_ascii=False),
            },
        ]
    )

    cv_summary = pd.DataFrame(rf_search.cv_results_).sort_values("rank_test_score").head(30)
    cv_summary["mean_test_rmse"] = -cv_summary["mean_test_score"]
    cv_summary["std_test_rmse"] = cv_summary["std_test_score"].abs()

    return (
        metrics,
        model_compare,
        cv_summary,
        y_test,
        y_pred_rf,
        X_test,
        best_rf,
    )


def build_feature_importance(
    model: RandomForestRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selected_cols: List[str],
) -> pd.DataFrame:
    impurity = model.feature_importances_
    perm = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    perm_mean = perm.importances_mean

    imp_df = pd.DataFrame(
        {
            "feature": selected_cols,
            "impurity_importance": impurity,
            "permutation_importance": perm_mean,
        }
    )

    def normalize(arr: pd.Series) -> pd.Series:
        max_v = float(arr.max())
        min_v = float(arr.min())
        if np.isclose(max_v, min_v):
            return pd.Series(np.zeros(len(arr)), index=arr.index)
        return (arr - min_v) / (max_v - min_v)

    imp_df["impurity_norm"] = normalize(imp_df["impurity_importance"])
    imp_df["permutation_norm"] = normalize(imp_df["permutation_importance"])
    imp_df["combined_score"] = (imp_df["impurity_norm"] + imp_df["permutation_norm"]) / 2.0
    imp_df = imp_df.sort_values("combined_score", ascending=False)
    top10 = imp_df.head(10).copy()
    top10.to_csv(os.path.join(DIR_RESULTS, "top10_features.csv"), index=False, encoding="utf-8-sig")
    return top10


def plot_outputs(y_true: np.ndarray, y_pred: np.ndarray, top10: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("真实 LN_IC50")
    plt.ylabel("预测 LN_IC50")
    plt.title("真实值 vs 预测值（随机森林）")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "true_vs_pred.png"), dpi=300)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("预测 LN_IC50")
    plt.ylabel("残差（真实 - 预测）")
    plt.title("残差图（随机森林）")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "residuals.png"), dpi=300)
    plt.close()

    plot_df = top10.iloc[::-1].copy()
    plt.figure(figsize=(8, 6))
    plt.barh(plot_df["feature"], plot_df["combined_score"])
    plt.xlabel("综合重要性分数")
    plt.ylabel("特征")
    plt.title("Top10 特征重要性（综合）")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_FIGURES, "top10_feature_importance.png"), dpi=300)
    plt.close()


def write_readmes(ctx: RunContext) -> None:
    readme_path = os.path.join(BASE_DIR, "README.md")
    run_path = os.path.join(BASE_DIR, "run_reproduce.md")

    readme = f"""# 随机森林复现项目说明

本目录基于本地 5 份原始数据文件复现“随机森林识别食管癌 ERBB2 靶向药物敏感性关键预测特征”的核心流程。

## 当前复现口径
- 药物范围：以 `results/candidate_drugs.csv` 的 HER2/ERBB 相关候选药物为准。
- 细胞系范围：以 `results/esophageal_cell_lines.csv` 的食管/GEJ 细胞系为主。
- 目标变量：`LN_IC50`（按自然对数尺度使用，不额外换算为 `log2(IC50)`）。
- 特征：全转录组表达 + 重点基因 + ERBB2 位点 CNV + 药物哑变量。

## 本次运行摘要
- 整合后真实样本数：{ctx.merged_rows}
- 整合后特征数：{ctx.merged_feature_cols}
- Bootstrap 样本数：{ctx.bootstrap_rows}
- 方差阈值后特征数：{ctx.selected_feature_cols}
- 重点基因缺失：{", ".join(ctx.missing_focus_genes) if ctx.missing_focus_genes else "无"}
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    run_md = """# 复现实验运行说明

## 运行环境
- Python 3.10+（本次实跑环境为 Python 3.13）
- 依赖：pandas, numpy, scikit-learn, matplotlib, openpyxl

## 执行命令
```powershell
& 'C:\\Users\\18210\\AppData\\Local\\Programs\\Python\\Python313\\python.exe' .\\src\\reproduce_core.py
```

## 主要输出
- `intermediate/merged_dataset.csv`
- `intermediate/bootstrap_dataset.csv`
- `results/metrics_summary.csv`
- `results/model_comparison.csv`
- `results/top10_features.csv`
- `results/matching_report.csv`
- `figures/true_vs_pred.png`
- `figures/residuals.png`
- `figures/top10_feature_importance.png`
"""
    with open(run_path, "w", encoding="utf-8") as f:
        f.write(run_md)


def main() -> None:
    ensure_dirs()
    start = datetime.now()
    print(f"[{start}] 开始执行第二阶段复现")

    gdsc, annot, candidate, eso = load_inputs()
    print(f"GDSC2 原始记录: {len(gdsc)}")
    print(f"候选药物数量: {candidate['DRUG_NAME'].nunique()}")
    print(f"食管/GEJ 细胞系数量: {len(eso)}")

    matching, gdsc_target = build_matching(gdsc, annot, candidate, eso)
    print(f"候选药物匹配后记录数: {len(matching)}")
    print(f"目标队列药敏记录数: {len(gdsc_target)}")

    sample_ids = sorted(gdsc_target["CCLE_ID"].dropna().astype(str).unique().tolist())
    print(f"目标队列唯一 CCLE_ID 数: {len(sample_ids)}")

    expr_t = read_gct_selected_samples(sample_ids)
    print(f"表达矩阵（样本x特征）: {expr_t.shape}")

    cnv = extract_erbb2_cnv()
    print(f"ERBB2 CNV 特征记录数: {len(cnv)}")

    merged, missing_focus = build_merged_dataset(gdsc_target, expr_t, cnv)
    print(f"merged_dataset 行数: {len(merged)} 列数: {len(merged.columns)}")

    _, boot, selected_cols, ctx = preprocess_and_bootstrap(merged)
    ctx.missing_focus_genes = missing_focus
    print(f"bootstrap_dataset 行数: {len(boot)}")
    print(f"方差阈值后特征数: {len(selected_cols)}")

    metrics, model_compare, cv_summary, y_test, y_pred_rf, X_test, best_rf = train_and_evaluate(
        boot, selected_cols
    )

    metrics["ln_ic50_note"] = ctx.ln_ic50_note
    metrics["python_version"] = os.sys.version.replace("\n", " ")
    metrics.to_csv(os.path.join(DIR_RESULTS, "metrics_summary.csv"), index=False, encoding="utf-8-sig")

    model_compare.to_csv(os.path.join(DIR_RESULTS, "model_comparison.csv"), index=False, encoding="utf-8-sig")
    cv_summary.to_csv(os.path.join(DIR_RESULTS, "rf_cv_summary_top30.csv"), index=False, encoding="utf-8-sig")

    top10 = build_feature_importance(best_rf, X_test, y_test, selected_cols)
    plot_outputs(y_test, y_pred_rf, top10)
    write_readmes(ctx)

    end = datetime.now()
    runtime_min = (end - start).total_seconds() / 60.0
    with open(os.path.join(DIR_LOGS, "reproduce_run.log"), "w", encoding="utf-8") as f:
        f.write(f"start={start.isoformat()}\n")
        f.write(f"end={end.isoformat()}\n")
        f.write(f"runtime_min={runtime_min:.2f}\n")
        f.write(f"merged_rows={ctx.merged_rows}\n")
        f.write(f"bootstrap_rows={ctx.bootstrap_rows}\n")
        f.write(f"selected_feature_cols={ctx.selected_feature_cols}\n")
        f.write(f"ln_ic50_note={ctx.ln_ic50_note}\n")

    print(f"[{end}] 完成。总耗时 {runtime_min:.2f} 分钟")
    print(f"图形输出: {os.path.join(DIR_FIGURES, 'true_vs_pred.png')}")
    print(f"图形输出: {os.path.join(DIR_FIGURES, 'residuals.png')}")
    print(f"图形输出: {os.path.join(DIR_FIGURES, 'top10_feature_importance.png')}")


if __name__ == "__main__":
    main()
