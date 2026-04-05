#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Improve Chinese rendering on Windows if available.
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_MERGED = os.path.join(BASE_DIR, "intermediate", "merged_dataset.csv")
IN_BOOT = os.path.join(BASE_DIR, "intermediate", "bootstrap_dataset.csv")
IN_MODEL_QUICK = os.path.join(BASE_DIR, "results_quick", "model_comparison.csv")
IN_METRIC_QUICK = os.path.join(BASE_DIR, "results_quick", "metrics_summary.csv")
IN_MATCH = os.path.join(BASE_DIR, "results", "matching_report.csv")
IN_BOOT_SUMMARY = os.path.join(BASE_DIR, "results", "bootstrap_distribution_summary.csv")

OUT_DIR = os.path.join(BASE_DIR, "paper_replace_outputs")
OUT_TABLE1 = os.path.join(OUT_DIR, "table1_new_model_performance.csv")
OUT_TABLE2 = os.path.join(OUT_DIR, "table2_new_top10_features.csv")
OUT_FIG3 = os.path.join(OUT_DIR, "figure3_new.png")
OUT_FIG4 = os.path.join(OUT_DIR, "figure4_new.png")
OUT_FIG1 = os.path.join(OUT_DIR, "figure1_new.png")
OUT_SUMMARY = os.path.join(OUT_DIR, "replacement_summary.md")

ART_PRED = os.path.join(OUT_DIR, "_artifacts_predictions.csv")
ART_IMP = os.path.join(OUT_DIR, "_artifacts_feature_importance.csv")
ART_CTX = os.path.join(OUT_DIR, "_artifacts_context.json")

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


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def _safe_float(v, nd=6):
    if pd.isna(v):
        return np.nan
    return round(float(v), nd)


def _load_merged() -> pd.DataFrame:
    if not os.path.exists(IN_MERGED):
        raise FileNotFoundError(f"缺少输入文件: {IN_MERGED}")
    return pd.read_csv(IN_MERGED, low_memory=False)


def _get_light_features(df: pd.DataFrame) -> List[str]:
    features: List[str] = []
    for g in FOCUS_GENES:
        col = f"EXP_{g}"
        if col in df.columns:
            features.append(col)
    for c in ["ERBB2_CNV_MODAL_TOTAL_CN", "ERBB2_CNV_LOH", "ERBB2_CNV_HOMDEL"]:
        if c in df.columns:
            features.append(c)

    drug_dummy_cols = [c for c in df.columns if c.startswith("DRUG_") and c not in {"DRUG_ID", "DRUG_NAME"}]
    if len(drug_dummy_cols) == 0 and "DRUG_NAME" in df.columns:
        dummies = pd.get_dummies(df["DRUG_NAME"], prefix="DRUG")
        for c in dummies.columns:
            df[c] = dummies[c]
        drug_dummy_cols = list(dummies.columns)
    features.extend(sorted(drug_dummy_cols))
    return sorted(list(dict.fromkeys(features)))


def prepare() -> str:
    ensure_dir()
    df = _load_merged()
    if "LN_IC50" not in df.columns:
        raise RuntimeError("merged_dataset.csv 缺少 LN_IC50 列。")

    feature_cols = _get_light_features(df)
    if len(feature_cols) == 0:
        raise RuntimeError("未能从 merged_dataset.csv 构建轻量特征。")

    data = df[feature_cols + ["LN_IC50", "DRUG_NAME", "CCLE_ID", "CELL_LINE_NAME"]].copy()
    data = data.dropna(subset=["LN_IC50"]).copy()
    n_total = len(data)
    if n_total < 10:
        raise RuntimeError(f"样本数过少（{n_total}），无法稳定生成替代表格和图。")

    test_size = 0.3
    random_state = 42
    rng = np.random.RandomState(random_state)
    test_idx = data.sample(frac=test_size, random_state=random_state).index
    train_idx = data.index.difference(test_idx)

    train = data.loc[train_idx].copy()
    test = data.loc[test_idx].copy()

    X_train_raw = train[feature_cols]
    X_test_raw = test[feature_cols]
    y_train = train["LN_IC50"].astype(float).to_numpy()
    y_test = test["LN_IC50"].astype(float).to_numpy()

    imp = SimpleImputer(strategy="mean")
    X_train = imp.fit_transform(X_train_raw)
    X_test = imp.transform(X_test_raw)

    rf = RandomForestRegressor(
        n_estimators=200,
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

    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    rf_r2 = r2_score(y_test, y_pred_rf)
    lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    lr_r2 = r2_score(y_test, y_pred_lr)

    svr_rmse = np.nan
    svr_r2 = np.nan
    svr_note = "按当前替代任务要求跳过SVR重新训练"
    if os.path.exists(IN_MODEL_QUICK):
        try:
            quick = pd.read_csv(IN_MODEL_QUICK)
            row = quick.loc[quick["model"] == "SVR_RBF"]
            if len(row) > 0:
                svr_rmse = _safe_float(row.iloc[0]["test_rmse"])
                svr_r2 = _safe_float(row.iloc[0]["test_r2"])
                svr_note = "复用 results_quick/model_comparison.csv 的现成SVR结果（非本次轻量重算）"
        except Exception:
            pass

    rows = [
        {
            "模型": "随机森林回归（轻量）",
            "目标变量": "LN_IC50",
            "RMSE": _safe_float(rf_rmse),
            "R2": _safe_float(rf_r2),
            "训练集占比": 0.7,
            "测试集占比": 0.3,
            "交叉验证折数": 0,
            "随机种子": random_state,
            "特征规模": len(feature_cols),
            "样本量": n_total,
            "备注": "固定参数 n_estimators=200, max_depth=8, min_samples_split=2；未做长时间网格搜索",
        },
        {
            "模型": "线性回归（轻量）",
            "目标变量": "LN_IC50",
            "RMSE": _safe_float(lr_rmse),
            "R2": _safe_float(lr_r2),
            "训练集占比": 0.7,
            "测试集占比": 0.3,
            "交叉验证折数": 0,
            "随机种子": random_state,
            "特征规模": len(feature_cols),
            "样本量": n_total,
            "备注": "最小成本基线模型，直接基于同一轻量特征集计算",
        },
        {
            "模型": "SVR（复用/未重训）",
            "目标变量": "LN_IC50",
            "RMSE": _safe_float(svr_rmse),
            "R2": _safe_float(svr_r2),
            "训练集占比": 0.7,
            "测试集占比": 0.3,
            "交叉验证折数": 5 if not pd.isna(svr_rmse) else 0,
            "随机种子": 42 if not pd.isna(svr_rmse) else random_state,
            "特征规模": len(feature_cols),
            "样本量": n_total,
            "备注": svr_note,
        },
    ]
    pd.DataFrame(rows).to_csv(OUT_TABLE1, index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_rf": y_pred_rf,
            "residual": y_test - y_pred_rf,
            "CCLE_ID": test["CCLE_ID"].values,
            "DRUG_NAME": test["DRUG_NAME"].values,
            "CELL_LINE_NAME": test["CELL_LINE_NAME"].values,
        }
    )
    pred_df.to_csv(ART_PRED, index=False, encoding="utf-8-sig")

    imp_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": rf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    imp_df.to_csv(ART_IMP, index=False, encoding="utf-8-sig")

    ctx = {
        "created_at": datetime.now().isoformat(),
        "input_files": {
            "merged_dataset": IN_MERGED,
            "bootstrap_dataset": IN_BOOT if os.path.exists(IN_BOOT) else None,
            "matching_report": IN_MATCH if os.path.exists(IN_MATCH) else None,
            "bootstrap_distribution_summary": IN_BOOT_SUMMARY if os.path.exists(IN_BOOT_SUMMARY) else None,
            "metrics_quick": IN_METRIC_QUICK if os.path.exists(IN_METRIC_QUICK) else None,
            "model_quick": IN_MODEL_QUICK if os.path.exists(IN_MODEL_QUICK) else None,
        },
        "model_setup": {
            "target": "LN_IC50",
            "train_ratio": 0.7,
            "test_ratio": 0.3,
            "random_state": random_state,
            "rf_params": {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_split": 2,
            },
            "svr_retrained": False,
            "permutation_importance": False,
            "bootstrap_rerun": False,
        },
        "feature_cols": feature_cols,
        "n_samples": n_total,
        "n_features": len(feature_cols),
        "rf_metrics": {"RMSE": _safe_float(rf_rmse), "R2": _safe_float(rf_r2)},
        "lr_metrics": {"RMSE": _safe_float(lr_rmse), "R2": _safe_float(lr_r2)},
    }
    with open(ART_CTX, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    return OUT_TABLE1


def table2() -> str:
    ensure_dir()
    if not os.path.exists(ART_IMP):
        raise FileNotFoundError("缺少轻量特征重要性中间件，请先执行 prepare。")
    imp = pd.read_csv(ART_IMP)
    top = imp.head(10).copy()

    def data_type(feature: str) -> str:
        if feature.startswith("EXP_"):
            return "基因表达"
        if feature.startswith("ERBB2_CNV_"):
            return "拷贝数"
        if feature.startswith("DRUG_"):
            return "药物指示变量"
        return "其他"

    annotation_map: Dict[str, str] = {
        "EXP_ERBB2": "HER2 受体酪氨酸激酶，ERBB2 通路核心基因",
        "EXP_GRB7": "ERBB2 邻近适配蛋白，参与下游信号转导",
        "EXP_ERBB3": "ERBB 家族成员，和 ERBB2 形成异源二聚体",
        "EXP_PIK3CA": "PI3K/AKT 通路关键催化亚基",
        "EXP_AKT1": "PI3K 下游关键激酶，调控存活/增殖",
        "EXP_MAPK1": "MAPK/ERK 通路核心激酶",
        "EXP_PTK6": "与上皮肿瘤信号调控相关的非受体激酶",
        "EXP_CCND1": "细胞周期调控基因，关联增殖活性",
        "EXP_SHC1": "受体下游衔接蛋白，参与 ERBB 信号传递",
        "ERBB2_CNV_MODAL_TOTAL_CN": "ERBB2 位点总拷贝数估计",
        "ERBB2_CNV_LOH": "ERBB2 位点杂合性缺失状态",
        "ERBB2_CNV_HOMDEL": "ERBB2 位点纯合缺失状态",
    }

    rows = []
    for i, r in enumerate(top.itertuples(index=False), start=1):
        feat = str(r.feature)
        short_name = feat.replace("EXP_", "")
        if feat.startswith("DRUG_"):
            short_name = feat.replace("DRUG_", "")
        ann = annotation_map.get(feat)
        if ann is None and feat.startswith("DRUG_"):
            ann = f"药物类别哑变量：{short_name}"
        if ann is None:
            ann = "基于当前数据可识别的统计学重要特征"
        rows.append(
            {
                "排名": i,
                "特征名称": short_name,
                "数据类型": data_type(feat),
                "生物学注释": ann,
                "重要性权重": _safe_float(r.importance, nd=8),
            }
        )

    pd.DataFrame(rows).to_csv(OUT_TABLE2, index=False, encoding="utf-8-sig")
    return OUT_TABLE2


def figure3() -> str:
    ensure_dir()
    if not os.path.exists(ART_PRED):
        raise FileNotFoundError("缺少预测中间件，请先执行 prepare。")
    pred = pd.read_csv(ART_PRED)
    if len(pred) == 0:
        raise RuntimeError("预测中间件为空，无法绘图。")

    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred_rf"].to_numpy(dtype=float)
    residual = pred["residual"].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    n = len(pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=220)
    ax1, ax2 = axes

    ax1.scatter(y_true, y_pred, c="#1f77b4", alpha=0.8, edgecolors="white", linewidths=0.5)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax1.plot([lo, hi], [lo, hi], "--", color="gray", linewidth=1.2)
    ax1.set_title("A. 真实值 vs 预测值", fontsize=12)
    ax1.set_xlabel("真实 LN_IC50")
    ax1.set_ylabel("预测 LN_IC50")
    ax1.text(
        0.02,
        0.98,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3f}\nn = {n}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        fontsize=10,
    )

    ax2.scatter(y_pred, residual, c="#d62728", alpha=0.8, edgecolors="white", linewidths=0.5)
    ax2.axhline(0.0, linestyle="--", color="gray", linewidth=1.2)
    ax2.set_title("B. 残差图", fontsize=12)
    ax2.set_xlabel("预测 LN_IC50")
    ax2.set_ylabel("残差（真实 - 预测）")
    ax2.text(
        0.02,
        0.98,
        f"残差均值 = {residual.mean():.3f}\n残差标准差 = {residual.std(ddof=1):.3f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        fontsize=10,
    )

    fig.suptitle("图3（替代版）：模型预测性能与残差分布", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT_FIG3, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG3


def figure4() -> str:
    ensure_dir()
    if not os.path.exists(OUT_TABLE2):
        raise FileNotFoundError("缺少 table2 输出，请先执行 table2。")
    t2 = pd.read_csv(OUT_TABLE2)
    if len(t2) == 0:
        raise RuntimeError("table2 为空，无法绘制图4。")

    plot_df = t2.sort_values("排名", ascending=True).copy()
    fig = plt.figure(figsize=(14, 7), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.barh(plot_df["特征名称"][::-1], plot_df["重要性权重"][::-1], color="#4c72b0")
    ax1.set_title("A. Top10 关键特征重要性", fontsize=12)
    ax1.set_xlabel("重要性权重（Impurity）")
    ax1.set_ylabel("特征")

    ax2.axis("off")
    table_text = plot_df[["排名", "特征名称", "数据类型"]].copy()
    table = ax2.table(
        cellText=table_text.values,
        colLabels=table_text.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    ax2.set_title("B. Top10 特征清单", fontsize=12, pad=12)

    fig.suptitle("图4（替代版）：关键特征重要性与特征类型", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG4, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG4


def figure1() -> str:
    ensure_dir()
    df = _load_merged()
    if "EXP_ERBB2" not in df.columns:
        raise RuntimeError("merged_dataset.csv 缺少 EXP_ERBB2，无法生成图1替代图。")
    sub = df[["DRUG_NAME", "EXP_ERBB2", "LN_IC50"]].copy()
    sub["EXP_ERBB2"] = pd.to_numeric(sub["EXP_ERBB2"], errors="coerce")
    sub["LN_IC50"] = pd.to_numeric(sub["LN_IC50"], errors="coerce")
    sub = sub.dropna(subset=["DRUG_NAME", "EXP_ERBB2", "LN_IC50"])
    if len(sub) < 8:
        raise RuntimeError("ERBB2 相关有效样本过少，无法稳定绘图。")

    drug_counts = sub["DRUG_NAME"].value_counts()
    drugs = drug_counts.index.tolist()[:2]
    drug1 = drugs[0]
    drug2 = drugs[1] if len(drugs) > 1 else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=220)

    def plot_scatter(ax, dname: str, title_prefix: str):
        d = sub[sub["DRUG_NAME"] == dname].copy()
        x = d["EXP_ERBB2"].to_numpy(dtype=float)
        y = d["LN_IC50"].to_numpy(dtype=float)
        ax.scatter(x, y, c="#2ca02c", alpha=0.8, edgecolors="white", linewidths=0.5)
        if len(d) >= 2:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color="black", linewidth=1.2)
            r = np.corrcoef(x, y)[0, 1]
            ax.text(
                0.03,
                0.95,
                f"r = {r:.3f}\nn = {len(d)}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
                fontsize=10,
            )
        ax.set_title(f"{title_prefix}. {dname}", fontsize=11)
        ax.set_xlabel("ERBB2 表达（RPKM）")
        ax.set_ylabel("LN_IC50")

    plot_scatter(axes[0], drug1, "A")

    if drug2 is not None:
        plot_scatter(axes[1], drug2, "B")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "仅检测到 1 个药物可用于相关分析", ha="center", va="center")

    axes[2].hist(sub["EXP_ERBB2"], bins=15, color="#1f77b4", edgecolor="white", alpha=0.9)
    axes[2].set_title("C. ERBB2 表达分布", fontsize=11)
    axes[2].set_xlabel("ERBB2 表达（RPKM）")
    axes[2].set_ylabel("频数")
    axes[2].text(
        0.03,
        0.95,
        f"n = {len(sub)}\n均值 = {sub['EXP_ERBB2'].mean():.2f}\n中位数 = {sub['EXP_ERBB2'].median():.2f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        fontsize=10,
    )

    fig.suptitle("图1（替代版）：ERBB2 表达与药敏相关性", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG1, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(ART_CTX):
        with open(ART_CTX, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    else:
        ctx = {}
    ctx["figure1_drugs"] = [drug1] + ([drug2] if drug2 is not None else [])
    with open(ART_CTX, "w", encoding="utf-8") as f:
        json.dump(ctx, f, ensure_ascii=False, indent=2)

    return OUT_FIG1


def summary() -> str:
    ensure_dir()
    if os.path.exists(ART_CTX):
        with open(ART_CTX, "r", encoding="utf-8") as f:
            ctx = json.load(f)
    else:
        ctx = {}

    t1 = pd.read_csv(OUT_TABLE1) if os.path.exists(OUT_TABLE1) else pd.DataFrame()
    t2 = pd.read_csv(OUT_TABLE2) if os.path.exists(OUT_TABLE2) else pd.DataFrame()

    lines = [
        "# 替代图表生成说明",
        "",
        "## 一、实际使用的数据文件",
        "",
        f"- `intermediate/merged_dataset.csv`：主分析数据来源（药敏 + 表达 + CNV + 药物信息）。",
        f"- `intermediate/bootstrap_dataset.csv`：仅作为可用性核对，不进行额外长时间重采样。",
        f"- `results/matching_report.csv`：用于说明样本匹配来源。",
        f"- `results/bootstrap_distribution_summary.csv`：用于说明已有Bootstrap摘要来源。",
        f"- `results_quick/model_comparison.csv`、`results_quick/metrics_summary.csv`：仅在SVR行可用时作现成复用参考。",
        "",
        "## 二、直接复用 vs 轻量重算",
        "",
        "- 直接复用：样本整合结果、匹配报告、Bootstrap摘要、已有quick模型文件（仅作参考）。",
        "- 轻量重算：使用 `merged_dataset.csv` 进行固定参数随机森林和线性回归，生成替代表1/表2以及替代图3/图4/图1。",
        "- 为保证时效，未执行完整网格搜索、未执行新增Bootstrap、未执行permutation importance、未执行SVR重训练。",
        "",
        "## 三、表1新数值来源",
        "",
        "- 随机森林与线性回归：本次轻量重算（7:3划分，random_state=42，目标变量 LN_IC50）。",
        "- SVR：未重训；如有数值则来自现有 `results_quick/model_comparison.csv` 的复用结果，并在备注中标注。",
        "",
        "## 四、表2新数值来源",
        "",
        "- 来自轻量随机森林的 impurity-based feature importance（Top10）。",
        "- 数据类型与生物学注释由特征前缀（EXP_/ERBB2_CNV_/DRUG_）和通路背景进行规则化标注。",
        "",
        "## 五、图3与图4数据来源",
        "",
        "- 图3：基于轻量随机森林测试集预测结果（`_artifacts_predictions.csv`）绘制真实值-预测值散点与残差图。",
        "- 图4：基于表2 Top10 特征权重绘制条形图并附特征清单，作为可替代论文图4的组合版。",
        "",
        "## 六、图1数据来源",
        "",
        "- 使用 `merged_dataset.csv` 中 ERBB2 表达（`EXP_ERBB2`）与 LN_IC50。",
        f"- 实际药物：{', '.join(ctx.get('figure1_drugs', [])) if ctx.get('figure1_drugs') else '未生成成功，无法确定药物'}。",
        "",
        "## 七、与论文原表/原图可能不一致的原因",
        "",
        "- 本次优先目标是快速替代可交付，采用轻量固定参数而非完整长时搜索。",
        "- 未新增permutation importance与SVR重训，特征排序与模型性能可能与原文不同。",
        "- 原论文可能使用了更复杂筛选规则、不同版本依赖或不同样本扩增策略。",
        "- 因采用当前目录现有中间件，最终数值反映“当前数据版本”的真实计算结果。",
    ]

    if len(t1) > 0:
        lines.extend(
            [
                "",
                "## 八、关键结果快照（表1）",
                "",
            ]
        )
        for _, r in t1.iterrows():
            lines.append(f"- {r['模型']}: RMSE={r['RMSE']}, R2={r['R2']}")
    if len(t2) > 0:
        lines.extend(
            [
                "",
                "## 九、关键结果快照（表2前3项）",
                "",
            ]
        )
        for _, r in t2.head(3).iterrows():
            lines.append(f"- 排名{int(r['排名'])}: {r['特征名称']}（权重={r['重要性权重']}）")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return OUT_SUMMARY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        choices=["prepare", "table2", "figure3", "figure4", "figure1", "summary"],
        required=True,
    )
    args = parser.parse_args()

    if args.step == "prepare":
        print(prepare())
    elif args.step == "table2":
        print(table2())
    elif args.step == "figure3":
        print(figure3())
    elif args.step == "figure4":
        print(figure4())
    elif args.step == "figure1":
        print(figure1())
    elif args.step == "summary":
        print(summary())


if __name__ == "__main__":
    main()
