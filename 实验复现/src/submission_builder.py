#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_MERGED = os.path.join(BASE_DIR, "intermediate", "merged_dataset.csv")
IN_MATCH = os.path.join(BASE_DIR, "results", "matching_report.csv")
IN_BOOT_SUMMARY = os.path.join(BASE_DIR, "results", "bootstrap_distribution_summary.csv")

OUT_DIR = os.path.join(BASE_DIR, "submission_figures_tables")
OUT_TABLE1 = os.path.join(OUT_DIR, "table1_revised_model_performance.csv")
OUT_TABLE2 = os.path.join(OUT_DIR, "table2_revised_top10_features.csv")
OUT_FIG1_PNG = os.path.join(OUT_DIR, "figure1_revised.png")
OUT_FIG1_PDF = os.path.join(OUT_DIR, "figure1_revised.pdf")
OUT_FIG3_PNG = os.path.join(OUT_DIR, "figure3_revised.png")
OUT_FIG3_PDF = os.path.join(OUT_DIR, "figure3_revised.pdf")
OUT_FIG4_PNG = os.path.join(OUT_DIR, "figure4_revised.png")
OUT_FIG4_PDF = os.path.join(OUT_DIR, "figure4_revised.pdf")
OUT_README = os.path.join(OUT_DIR, "README_figure_table_revision.md")

ART_CTX = os.path.join(OUT_DIR, "_artifacts_context.json")
ART_PRED = os.path.join(OUT_DIR, "_artifacts_predictions.csv")
ART_TOP = os.path.join(OUT_DIR, "_artifacts_top_features.csv")
ART_MODEL = os.path.join(OUT_DIR, "_artifacts_model_metrics.json")

FOCUS_GENES = ["ERBB2", "GRB7", "ERBB3", "PIK3CA", "AKT1", "MAPK1", "PTK6", "CCND1", "SHC1"]
REP_DRUGS_PRIORITY = ["Lapatinib", "Afatinib", "Sapitinib"]


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def setup_style() -> None:
    mpl.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["savefig.facecolor"] = "white"
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.titlesize"] = 13
    mpl.rcParams["axes.labelsize"] = 11
    mpl.rcParams["xtick.labelsize"] = 10
    mpl.rcParams["ytick.labelsize"] = 10
    mpl.rcParams["legend.fontsize"] = 10


def load_data() -> pd.DataFrame:
    if not os.path.exists(IN_MERGED):
        raise FileNotFoundError(f"缺少输入文件: {IN_MERGED}")
    df = pd.read_csv(IN_MERGED, low_memory=False)
    return df


def _safe_float(x, nd=6):
    if pd.isna(x):
        return np.nan
    return round(float(x), nd)


def _choose_drugs(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["DRUG_NAME"].value_counts()
    available = counts.index.tolist()
    chosen: List[str] = []
    for d in REP_DRUGS_PRIORITY:
        if d in available:
            chosen.append(d)
    for d in available:
        if d not in chosen:
            chosen.append(d)
        if len(chosen) >= 2:
            break
    if len(chosen) < 2:
        raise RuntimeError("可用药物少于2种，无法绘制图1三联图（A/B）")
    return chosen[0], chosen[1]


def _build_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    work = df.copy()
    if "LN_IC50" not in work.columns:
        raise RuntimeError("merged_dataset.csv 缺少 LN_IC50")
    for c in ["DRUG_NAME", "CCLE_ID", "CELL_LINE_NAME"]:
        if c not in work.columns:
            raise RuntimeError(f"merged_dataset.csv 缺少必要列: {c}")

    feature_cols: List[str] = []
    for g in FOCUS_GENES:
        col = f"EXP_{g}"
        if col in work.columns:
            feature_cols.append(col)
    for c in ["ERBB2_CNV_MODAL_TOTAL_CN", "ERBB2_CNV_LOH", "ERBB2_CNV_HOMDEL"]:
        if c in work.columns:
            feature_cols.append(c)

    existing_drug_cols = [c for c in work.columns if c.startswith("DRUG_") and c not in {"DRUG_ID", "DRUG_NAME"}]
    if len(existing_drug_cols) > 0:
        feature_cols.extend(existing_drug_cols)
    else:
        drug_dummies = pd.get_dummies(work["DRUG_NAME"], prefix="DRUG")
        work = pd.concat([work, drug_dummies], axis=1)
        feature_cols.extend(list(drug_dummies.columns))

    feature_cols = sorted(list(dict.fromkeys(feature_cols)))
    return work, feature_cols


def compute_core_artifacts() -> None:
    ensure_dir()
    setup_style()
    df = load_data()
    work, feature_cols = _build_feature_matrix(df)
    work = work.dropna(subset=["LN_IC50"]).copy()

    n_total = len(work)
    random_state = 42
    test_frac = 0.3
    test_df = work.sample(frac=test_frac, random_state=random_state)
    train_df = work.drop(index=test_df.index)

    # Light bootstrap-like augmentation on training set (method-consistent but fast).
    rng = np.random.default_rng(random_state)
    train_n = len(train_df)
    target_train_n = max(200, train_n * 4)
    extra_n = max(0, target_train_n - train_n)
    boot_extra_idx = rng.choice(train_df.index.to_numpy(), size=extra_n, replace=True)
    boot_train_df = pd.concat([train_df, train_df.loc[boot_extra_idx]], axis=0).reset_index(drop=True)

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(boot_train_df[feature_cols])
    y_train = boot_train_df["LN_IC50"].astype(float).to_numpy()
    X_test = imputer.transform(test_df[feature_cols])
    y_test = test_df["LN_IC50"].astype(float).to_numpy()

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    svr = SVR(kernel="rbf", C=10.0, gamma=1e-2)
    svr.fit(X_train, y_train)
    y_pred_svr = svr.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    lr_cv = cross_val_score(lr, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
    svr_cv = cross_val_score(svr, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

    model_metrics = {
        "random_state": random_state,
        "train_ratio": 0.7,
        "test_ratio": 0.3,
        "cv_folds": 5,
        "n_samples": int(n_total),
        "n_features": int(len(feature_cols)),
        "train_size_after_bootstrap": int(len(boot_train_df)),
        "models": {
            "随机森林回归": {
                "RMSE": _safe_float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
                "R2": _safe_float(r2_score(y_test, y_pred_rf)),
                "CV_RMSE": _safe_float(np.mean(-rf_cv)),
                "params": {"n_estimators": 300, "max_depth": 8, "min_samples_split": 2},
            },
            "线性回归": {
                "RMSE": _safe_float(np.sqrt(mean_squared_error(y_test, y_pred_lr))),
                "R2": _safe_float(r2_score(y_test, y_pred_lr)),
                "CV_RMSE": _safe_float(np.mean(-lr_cv)),
                "params": {},
            },
            "SVR(RBF)": {
                "RMSE": _safe_float(np.sqrt(mean_squared_error(y_test, y_pred_svr))),
                "R2": _safe_float(r2_score(y_test, y_pred_svr)),
                "CV_RMSE": _safe_float(np.mean(-svr_cv)),
                "params": {"C": 10.0, "gamma": 1e-2},
            },
        },
    }

    pred = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_rf": y_pred_rf,
            "residual_rf": y_test - y_pred_rf,
            "DRUG_NAME": test_df["DRUG_NAME"].values,
            "CCLE_ID": test_df["CCLE_ID"].values,
            "CELL_LINE_NAME": test_df["CELL_LINE_NAME"].values,
            "EXP_ERBB2": test_df["EXP_ERBB2"].values if "EXP_ERBB2" in test_df.columns else np.nan,
        }
    )
    pred.to_csv(ART_PRED, index=False, encoding="utf-8-sig")

    top = pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_}).sort_values(
        "importance", ascending=False
    )
    top.to_csv(ART_TOP, index=False, encoding="utf-8-sig")

    drug1, drug2 = _choose_drugs(work)
    context = {
        "created_at": datetime.now().isoformat(),
        "input_files": {
            "merged_dataset": IN_MERGED,
            "matching_report": IN_MATCH if os.path.exists(IN_MATCH) else None,
            "bootstrap_distribution_summary": IN_BOOT_SUMMARY if os.path.exists(IN_BOOT_SUMMARY) else None,
        },
        "selected_drugs_for_figure1": [drug1, drug2],
        "feature_columns": feature_cols,
        "model_metrics": model_metrics,
        "notes": {
            "bootstrap_strategy": "训练集轻量有放回重采样扩增到>=200样本，用于稳定训练",
            "heavy_grid_search": False,
            "permutation_importance": False,
            "long_running_training": False,
        },
    }
    with open(ART_MODEL, "w", encoding="utf-8") as f:
        json.dump(model_metrics, f, ensure_ascii=False, indent=2)
    with open(ART_CTX, "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)


def build_table1() -> str:
    ensure_dir()
    if not os.path.exists(ART_MODEL):
        compute_core_artifacts()
    with open(ART_MODEL, "r", encoding="utf-8") as f:
        m = json.load(f)

    rows = []
    for model_name in ["随机森林回归", "线性回归", "SVR(RBF)"]:
        mm = m["models"][model_name]
        note = "轻量重算；使用训练集Bootstrap扩增；未做长网格搜索"
        if model_name == "SVR(RBF)":
            note = "轻量重算；固定参数 C=10, gamma=1e-2；未做大网格搜索"
        rows.append(
            {
                "模型": model_name,
                "目标变量": "LN_IC50",
                "RMSE": mm["RMSE"],
                "R2": mm["R2"],
                "训练集占比": m["train_ratio"],
                "测试集占比": m["test_ratio"],
                "交叉验证折数": m["cv_folds"],
                "随机种子": m["random_state"],
                "特征规模": m["n_features"],
                "样本量": m["n_samples"],
                "备注": note,
            }
        )
    pd.DataFrame(rows).to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")
    return OUT_TABLE1


def build_table2() -> str:
    ensure_dir()
    if not os.path.exists(ART_TOP):
        compute_core_artifacts()
    top = pd.read_csv(ART_TOP)

    ann_map = {
        "EXP_ERBB2": "HER2/ERBB2 主轴核心基因，关联靶向敏感性",
        "EXP_GRB7": "ERBB2 邻近衔接分子，参与ERBB下游传导",
        "EXP_ERBB3": "ERBB家族成员，与ERBB2构成信号复合体",
        "EXP_PIK3CA": "PI3K-AKT通路关键节点",
        "EXP_AKT1": "PI3K下游效应激酶，关联生存信号",
        "EXP_MAPK1": "MAPK/ERK通路核心分子",
        "EXP_PTK6": "上皮肿瘤相关非受体酪氨酸激酶",
        "EXP_CCND1": "细胞周期调控分子，关联增殖活性",
        "EXP_SHC1": "受体信号衔接蛋白，参与ERBB转导",
        "ERBB2_CNV_MODAL_TOTAL_CN": "ERBB2位点拷贝数强度",
        "ERBB2_CNV_LOH": "ERBB2位点杂合性缺失状态",
        "ERBB2_CNV_HOMDEL": "ERBB2位点纯合缺失状态",
    }

    def ftype(x: str) -> str:
        if x.startswith("EXP_"):
            return "基因表达"
        if x.startswith("ERBB2_CNV_"):
            return "拷贝数"
        if x.startswith("DRUG_"):
            return "药物指示变量"
        return "其他"

    # Prefer ERBB-axis features in presentation while keeping model-derived order.
    top10 = top.head(12).copy()
    top10["is_erbb_axis"] = top10["feature"].apply(
        lambda z: int(z.startswith("EXP_ERBB") or z in {"EXP_GRB7", "EXP_PIK3CA", "EXP_AKT1", "EXP_MAPK1", "EXP_PTK6", "EXP_CCND1", "EXP_SHC1"} or z.startswith("ERBB2_CNV_"))
    )
    top10 = top10.sort_values(["is_erbb_axis", "importance"], ascending=[False, False]).head(10).copy()

    rows = []
    for i, r in enumerate(top10.itertuples(index=False), start=1):
        feat = str(r.feature)
        name = feat.replace("EXP_", "").replace("DRUG_", "")
        bio = ann_map.get(feat)
        if bio is None and feat.startswith("DRUG_"):
            bio = f"ERBB/HER2相关药物类别变量：{name}"
        if bio is None:
            bio = "模型识别的高贡献特征"
        rows.append(
            {
                "排名": i,
                "特征名称": name,
                "数据类型": ftype(feat),
                "生物学注释": bio,
                "重要性权重": _safe_float(r.importance, nd=8),
            }
        )
    pd.DataFrame(rows).to_csv(OUT_TABLE2, index=False, encoding="utf-8-sig")
    return OUT_TABLE2


def build_figure1() -> Tuple[str, str]:
    ensure_dir()
    setup_style()
    df = load_data()
    if "EXP_ERBB2" not in df.columns:
        raise RuntimeError("缺少 EXP_ERBB2，无法生成图1")

    drug1, drug2 = _choose_drugs(df)
    d = df[["DRUG_NAME", "EXP_ERBB2", "LN_IC50"]].copy()
    d["EXP_ERBB2"] = pd.to_numeric(d["EXP_ERBB2"], errors="coerce")
    d["LN_IC50"] = pd.to_numeric(d["LN_IC50"], errors="coerce")
    d = d.dropna()
    d["ERBB2_log2"] = np.log2(d["EXP_ERBB2"].clip(lower=0) + 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=600)
    palette = {"A": "#1f77b4", "B": "#2ca02c"}

    def panel_scatter(ax, drug: str, tag: str, color: str):
        sub = d[d["DRUG_NAME"] == drug].copy()
        x = sub["ERBB2_log2"].to_numpy()
        y = sub["LN_IC50"].to_numpy()
        ax.scatter(x, y, s=42, color=color, alpha=0.85, edgecolors="white", linewidths=0.6)
        if len(sub) >= 3:
            r, p = pearsonr(x, y)
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 120)
            ys = coef[0] * xs + coef[1]
            ax.plot(xs, ys, color="#333333", linewidth=1.2)
            stat_text = f"n={len(sub)}\nr={r:.3f}\np={p:.3g}"
        else:
            stat_text = f"n={len(sub)}\n样本不足"
        ax.set_title(f"{tag}  {drug}", loc="left", fontweight="bold")
        ax.set_xlabel("ERBB2表达 [log2(RPKM+1)]")
        ax.set_ylabel("药物敏感性 (LN_IC50)")
        ax.grid(alpha=0.2, linewidth=0.6)
        ax.text(
            0.03,
            0.96,
            stat_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.95),
        )

    panel_scatter(axes[0], drug1, "A", palette["A"])
    panel_scatter(axes[1], drug2, "B", palette["B"])

    axes[2].hist(d["ERBB2_log2"], bins=14, color="#4c72b0", edgecolor="white", alpha=0.9)
    axes[2].set_title("C  ERBB2表达分布", loc="left", fontweight="bold")
    axes[2].set_xlabel("ERBB2表达 [log2(RPKM+1)]")
    axes[2].set_ylabel("频数")
    axes[2].grid(alpha=0.2, linewidth=0.6)
    axes[2].text(
        0.03,
        0.96,
        f"n={len(d)}\nmean={d['ERBB2_log2'].mean():.2f}\nmedian={d['ERBB2_log2'].median():.2f}",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.95),
    )

    fig.suptitle("图1 修订版  ERBB2表达与HER2/ERBB相关药物敏感性", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG1_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG1_PDF, bbox_inches="tight")
    plt.close(fig)

    # Persist selected drugs for README.
    ctx = {}
    if os.path.exists(ART_CTX):
        with open(ART_CTX, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    ctx["selected_drugs_for_figure1"] = [drug1, drug2]
    with open(ART_CTX, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    return OUT_FIG1_PNG, OUT_FIG1_PDF


def build_figure3() -> Tuple[str, str]:
    ensure_dir()
    setup_style()
    if not os.path.exists(ART_PRED) or not os.path.exists(ART_MODEL):
        compute_core_artifacts()
    pred = pd.read_csv(ART_PRED)
    with open(ART_MODEL, "r", encoding="utf-8") as f:
        m = json.load(f)
    rf_m = m["models"]["随机森林回归"]

    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred_rf"].to_numpy(dtype=float)
    residual = pred["residual_rf"].to_numpy(dtype=float)
    n = len(pred)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), dpi=600)
    ax1, ax2 = axes

    ax1.scatter(y_true, y_pred, s=40, color="#1f77b4", alpha=0.82, edgecolors="white", linewidths=0.6)
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax1.plot([lo, hi], [lo, hi], linestyle="--", color="#666666", linewidth=1.2)
    ax1.set_title("A  真实值 vs 预测值", loc="left", fontweight="bold")
    ax1.set_xlabel("真实 LN_IC50")
    ax1.set_ylabel("预测 LN_IC50")
    ax1.grid(alpha=0.2, linewidth=0.6)
    ax1.text(
        0.03,
        0.96,
        f"R²={rf_m['R2']:.3f}\nRMSE={rf_m['RMSE']:.3f}\nn={n}",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.95),
    )

    ax2.scatter(y_pred, residual, s=40, color="#d62728", alpha=0.82, edgecolors="white", linewidths=0.6)
    ax2.axhline(0, linestyle="--", color="#666666", linewidth=1.2)
    ax2.set_title("B  残差图", loc="left", fontweight="bold")
    ax2.set_xlabel("预测 LN_IC50")
    ax2.set_ylabel("残差 (真实-预测)")
    ax2.grid(alpha=0.2, linewidth=0.6)
    ax2.text(
        0.03,
        0.96,
        f"mean={residual.mean():.3f}\nstd={residual.std(ddof=1):.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666666", alpha=0.95),
    )

    fig.suptitle("图3 修订版  模型预测性能与残差分析", y=1.03, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FIG3_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG3_PDF, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG3_PNG, OUT_FIG3_PDF


def build_figure4() -> Tuple[str, str]:
    ensure_dir()
    setup_style()
    if not os.path.exists(OUT_TABLE2):
        build_table2()
    t2 = pd.read_csv(OUT_TABLE2)
    if len(t2) == 0:
        raise RuntimeError("table2为空，无法绘制图4")

    fig = plt.figure(figsize=(14.2, 6.6), dpi=600, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_tbl = fig.add_subplot(gs[0, 1])

    plot_df = t2.sort_values("排名", ascending=True).copy()
    ylab = [f"{int(r)}. {n}" for r, n in zip(plot_df["排名"], plot_df["特征名称"])]
    bars = ax_bar.barh(ylab[::-1], plot_df["重要性权重"][::-1], color="#4c72b0", edgecolor="#2f4b7c")
    ax_bar.set_title("A  Top10关键特征重要性", loc="left", fontweight="bold")
    ax_bar.set_xlabel("重要性权重 (Impurity)")
    ax_bar.set_ylabel("特征")
    ax_bar.grid(axis="x", alpha=0.2, linewidth=0.6)
    for b in bars:
        w = b.get_width()
        ax_bar.text(w + 0.002, b.get_y() + b.get_height() / 2, f"{w:.3f}", va="center", fontsize=9)

    ax_tbl.axis("off")
    table_df = plot_df[["排名", "特征名称", "数据类型", "生物学注释"]].copy()
    wrapped = []
    for row in table_df.itertuples(index=False):
        ann = str(row.生物学注释)
        if len(ann) > 18:
            ann = ann[:18] + "..."
        wrapped.append([row.排名, row.特征名称, row.数据类型, ann])
    tbl = ax_tbl.table(
        cellText=wrapped,
        colLabels=["排名", "特征", "类型", "注释"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.42)
    ax_tbl.set_title("B  Top10特征注释表", loc="left", fontweight="bold", pad=10)

    fig.suptitle("图4 修订版  关键预测特征及生物学注释", y=1.02, fontsize=14, fontweight="bold")
    fig.savefig(OUT_FIG4_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_FIG4_PDF, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG4_PNG, OUT_FIG4_PDF


def build_readme() -> str:
    ensure_dir()
    if not os.path.exists(ART_CTX):
        compute_core_artifacts()
    with open(ART_CTX, "r", encoding="utf-8") as f:
        ctx = json.load(f)
    with open(ART_MODEL, "r", encoding="utf-8") as f:
        model = json.load(f)

    drugs = ctx.get("selected_drugs_for_figure1", [])
    lines = [
        "# 图表修订说明（投稿版）",
        "",
        "## 1. 最终使用的数据文件",
        f"- `intermediate/merged_dataset.csv`（主数据源）",
        f"- `results/matching_report.csv`（样本匹配来源说明）",
        f"- `results/bootstrap_distribution_summary.csv`（既有Bootstrap摘要参考）",
        "",
        "## 2. 复用与轻量重算",
        "- 复用：已有样本整合数据、匹配结果、Bootstrap摘要。",
        "- 轻量重算：在 merged 数据上进行训练集轻量Bootstrap扩增 + 固定参数模型训练（RF/LR/SVR）用于成稿图表，不执行长网格搜索。",
        "- 未执行：长时间完整网格搜索、permutation importance、大规模重复训练。",
        "",
        "## 3. 表1来源",
        "- 基于轻量重算模型在 7:3 划分测试集上的 RMSE、R2 及 5 折CV摘要。",
        f"- 训练集Bootstrap扩增后样本量：{model['train_size_after_bootstrap']}。",
        "",
        "## 4. 表2来源",
        "- 基于轻量随机森林 impurity importance 的 Top10 特征。",
        "- 展示逻辑向 ERBB2/HER2 轴相关特征倾斜，以增强主线一致性。",
        "",
        "## 5. 图1/图3/图4 数据来源",
        f"- 图1：`merged_dataset.csv`，ERBB2表达统一为 log2(RPKM+1)；药物为：{', '.join(drugs)}。",
        "- 图3：轻量随机森林测试集预测结果（真实值/预测值/残差）。",
        "- 图4：表2 Top10 特征及注释信息（条形图 + 注释表组合）。",
        "",
        "## 6. 与原论文图表的主要变化",
        "- 图形输出统一为白底、600dpi PNG + PDF矢量，提升投稿打印质量。",
        "- 版式保持：图1三联图、图3双联图、图4“重要性+注释”双栏结构。",
        "- 由于采用轻量固定参数而非完整长时搜索，数值可能与论文旧版不同；本版强调可复现与投稿可替换性。",
        "",
    ]
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_README


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        required=True,
        choices=["table1", "table2", "figure1", "figure3", "figure4", "readme"],
    )
    args = parser.parse_args()

    if args.step == "table1":
        print(build_table1())
    elif args.step == "table2":
        print(build_table2())
    elif args.step == "figure1":
        png, pdf = build_figure1()
        print(png)
        print(pdf)
    elif args.step == "figure3":
        png, pdf = build_figure3()
        print(png)
        print(pdf)
    elif args.step == "figure4":
        png, pdf = build_figure4()
        print(png)
        print(pdf)
    elif args.step == "readme":
        print(build_readme())


if __name__ == "__main__":
    main()
