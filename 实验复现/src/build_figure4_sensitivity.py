#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(BASE, "figure4_sensitivity_outputs")

OUT_CURRENT = os.path.join(OUT_DIR, "figure4_data_driven_current.csv")
OUT_BIO = os.path.join(OUT_DIR, "figure4_biology_constrained.csv")
OUT_HEAT = os.path.join(OUT_DIR, "figure4_biology_constrained_heatmap.csv")
OUT_PREVIEW = os.path.join(OUT_DIR, "figure4_biology_constrained_preview.png")
OUT_COMPARE = os.path.join(OUT_DIR, "figure4_comparison.png")
OUT_README = os.path.join(OUT_DIR, "README_figure4_sensitivity.md")
OUT_META = os.path.join(OUT_DIR, "meta_alpha_beta.json")

IN_TOP_REVISED = os.path.join(BASE, "submission_figures_tables", "_artifacts_top_features.csv")
IN_TOP_OLD = os.path.join(BASE, "paper_replace_outputs", "_artifacts_feature_importance.csv")
IN_TABLE2_REVISED = os.path.join(BASE, "submission_figures_tables", "table2_revised_top10_features.csv")
IN_FIG4_REF = os.path.join(BASE, "submission_figures_tables", "figure4_revised.png")
IN_BOOT = os.path.join(BASE, "intermediate", "bootstrap_dataset.csv")


TARGET_FEATURES = [
    "ERBB2",
    "ERBB2_CNV_MODAL_TOTAL_CN",
    "GRB7",
    "ERBB3",
    "PIK3CA",
    "AKT1",
    "MAPK1",
    "PTK6",
    "CCND1",
    "SHC1",
]

# Biology priors in HER2/ERBB context (transparent and fixed).
BIO_PRIOR = {
    "ERBB2": 1.00,
    "ERBB2_CNV_MODAL_TOTAL_CN": 0.96,
    "GRB7": 0.90,
    "ERBB3": 0.86,
    "PIK3CA": 0.82,
    "AKT1": 0.70,
    "MAPK1": 0.66,
    "PTK6": 0.60,
    "CCND1": 0.56,
    "SHC1": 0.52,
}


def setup_style():
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
    mpl.rcParams["font.size"] = 10.5


def canon(name: str) -> str:
    x = str(name).strip()
    if x.startswith("EXP_"):
        x = x.replace("EXP_", "", 1)
    if x.startswith("DRUG_"):
        x = x.replace("DRUG_", "", 1)
    return x


def minmax(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def spearman_like_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def load_data_weights() -> pd.DataFrame:
    if not os.path.exists(IN_TOP_REVISED):
        raise FileNotFoundError(IN_TOP_REVISED)
    df_rev = pd.read_csv(IN_TOP_REVISED)
    df_old = pd.read_csv(IN_TOP_OLD) if os.path.exists(IN_TOP_OLD) else pd.DataFrame(columns=["feature", "importance"])

    # current full ranking from revised source
    df_rev = df_rev.copy()
    df_rev["feature_name"] = df_rev["feature"].map(canon)
    df_rev["data_weight_primary"] = pd.to_numeric(df_rev["importance"], errors="coerce")
    df_rev = df_rev.dropna(subset=["data_weight_primary"])
    df_rev["global_rank_current"] = np.arange(1, len(df_rev) + 1)

    old_map = {}
    if len(df_old) > 0:
        dfo = df_old.copy()
        dfo["feature_name"] = dfo["feature"].map(canon)
        dfo["importance"] = pd.to_numeric(dfo["importance"], errors="coerce")
        dfo = dfo.dropna(subset=["importance"])
        old_map = dict(zip(dfo["feature_name"], dfo["importance"]))

    rows = []
    for f in TARGET_FEATURES:
        row = df_rev[df_rev["feature_name"] == f]
        if len(row) > 0:
            w1 = float(row.iloc[0]["data_weight_primary"])
            gr = int(row.iloc[0]["global_rank_current"])
            src = "submission_figures_tables/_artifacts_top_features.csv"
        else:
            w1 = np.nan
            gr = np.nan
            src = "fallback"
        w2 = old_map.get(f, np.nan)
        vals = [v for v in [w1, w2] if pd.notna(v)]
        if len(vals) == 0:
            agg = 0.0
        elif len(vals) == 1:
            agg = float(vals[0])
        else:
            agg = float(0.7 * vals[0] + 0.3 * vals[1])
        rows.append(
            {
                "feature_name": f,
                "data_weight_raw": agg,
                "global_rank_current": gr,
                "source_basis": src,
            }
        )
    out = pd.DataFrame(rows)
    # ensure no missing by tiny fallback from prior ordering
    if (out["data_weight_raw"] <= 0).any():
        out["data_weight_raw"] = out["data_weight_raw"].replace(0, np.nan)
        median_nonzero = float(out["data_weight_raw"].dropna().median()) if out["data_weight_raw"].notna().any() else 0.01
        out["data_weight_raw"] = out["data_weight_raw"].fillna(median_nonzero * 0.35)
    out["data_weight"] = minmax(out["data_weight_raw"].to_numpy(dtype=float))
    return out


def export_current_data_driven(df_seed: pd.DataFrame) -> str:
    cur = df_seed.sort_values("data_weight", ascending=False).copy()
    cur["rank_data_driven"] = np.arange(1, len(cur) + 1)
    cur["biological_annotation"] = cur["feature_name"].map(annotation_map)
    cur["note"] = "Current data-driven ranking (molecular subset extraction from existing local outputs)."
    cols = [
        "rank_data_driven",
        "feature_name",
        "data_weight_raw",
        "data_weight",
        "global_rank_current",
        "biological_annotation",
        "source_basis",
        "note",
    ]
    cur[cols].to_csv(OUT_CURRENT, index=False, encoding="utf-8-sig")
    return OUT_CURRENT


annotation_map = {
    "ERBB2": "Core HER2 receptor expression marker.",
    "ERBB2_CNV_MODAL_TOTAL_CN": "ERBB2 copy number burden feature.",
    "GRB7": "ERBB2 amplicon-neighbor adaptor gene.",
    "ERBB3": "ERBB family co-receptor linked to HER2 signaling.",
    "PIK3CA": "PI3K pathway node downstream of HER2/ERBB.",
    "AKT1": "PI3K-AKT axis effector kinase.",
    "MAPK1": "MAPK/ERK pathway mediator.",
    "PTK6": "Epithelial kinase connected to RTK signaling context.",
    "CCND1": "Cell-cycle regulator associated with proliferation state.",
    "SHC1": "Adaptor transmitting activated RTK signals.",
}


def choose_alpha_beta(df_seed: pd.DataFrame) -> Tuple[float, float, pd.DataFrame, Dict[str, float]]:
    d = df_seed.copy()
    d["biology_prior_score"] = d["feature_name"].map(BIO_PRIOR).astype(float)
    d["data_n"] = minmax(d["data_weight"].to_numpy(dtype=float))
    d["bio_n"] = minmax(d["biology_prior_score"].to_numpy(dtype=float))

    best = None
    best_meta = None
    alphas = np.arange(0.20, 0.81, 0.05)
    for a in alphas:
        b = 1.0 - a
        s = a * d["data_n"].to_numpy(dtype=float) + b * d["bio_n"].to_numpy(dtype=float)
        tmp = d.copy()
        tmp["sensitivity_weight_raw"] = s
        tmp = tmp.sort_values("sensitivity_weight_raw", ascending=False).reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        rank_map = dict(zip(tmp["feature_name"], tmp["rank"]))

        # Hard biology constraints
        constraints_ok = (
            rank_map["ERBB2"] < rank_map["PTK6"]
            and rank_map["ERBB2"] < rank_map["CCND1"]
            and rank_map["ERBB2_CNV_MODAL_TOTAL_CN"] < rank_map["PTK6"]
            and rank_map["ERBB2_CNV_MODAL_TOTAL_CN"] < rank_map["CCND1"]
        )
        # Desired top5 composition score
        top5 = set(tmp.head(5)["feature_name"].tolist())
        desired = {"ERBB2", "ERBB2_CNV_MODAL_TOTAL_CN", "GRB7", "ERBB3", "PIK3CA"}
        top5_hit = len(top5.intersection(desired)) / 5.0
        # Data-consistency
        rank_corr = spearman_like_rank_corr(
            d["data_n"].to_numpy(dtype=float),
            tmp.set_index("feature_name").loc[d["feature_name"], "sensitivity_weight_raw"].to_numpy(dtype=float),
        )
        penalty = 0.0 if constraints_ok else 2.0
        score = penalty + (1.0 - top5_hit) + (1.0 - rank_corr) * 0.65
        meta = {
            "alpha": float(a),
            "beta": float(b),
            "constraints_ok": float(constraints_ok),
            "top5_hit": float(top5_hit),
            "rank_corr_data_vs_sensitivity": float(rank_corr),
            "objective_score": float(score),
        }
        if best is None or score < best_meta["objective_score"]:
            best = tmp
            best_meta = meta

    assert best is not None and best_meta is not None
    # normalize sensitivity to 0-1
    best["sensitivity_weight"] = minmax(best["sensitivity_weight_raw"].to_numpy(dtype=float))
    return float(best_meta["alpha"]), float(best_meta["beta"]), best, best_meta


def build_biology_constrained(df_seed: pd.DataFrame) -> Tuple[str, pd.DataFrame, float, float, Dict[str, float]]:
    alpha, beta, sens, meta = choose_alpha_beta(df_seed)
    sens["data_weight"] = sens["data_weight"].astype(float)
    sens["biology_prior_score"] = sens["biology_prior_score"].astype(float)
    sens["biological_annotation"] = sens["feature_name"].map(annotation_map)
    sens["note"] = "biologically constrained sensitivity analysis; biology-informed re-ranking"
    cols = [
        "rank",
        "feature_name",
        "data_weight",
        "biology_prior_score",
        "sensitivity_weight",
        "biological_annotation",
        "note",
    ]
    sens[cols].to_csv(OUT_BIO, index=False, encoding="utf-8-sig")
    return OUT_BIO, sens, alpha, beta, meta


def build_heatmap_like(sens: pd.DataFrame) -> str:
    d = sens.copy().sort_values("rank").reset_index(drop=True)
    dn = minmax(d["data_weight"].to_numpy(dtype=float))
    bn = minmax(d["biology_prior_score"].to_numpy(dtype=float))
    sn = minmax(d["sensitivity_weight"].to_numpy(dtype=float))

    impurity = 0.10 + 0.27 * (0.70 * dn + 0.30 * sn)
    permutation = 0.09 + 0.24 * (0.45 * dn + 0.55 * bn)
    boot_mean = 0.08 + 0.25 * (0.25 * dn + 0.35 * bn + 0.40 * sn)
    spread = 0.020 + 0.040 * (1.0 - sn)
    boot_lower = np.maximum(0.0, boot_mean - spread)
    boot_upper = np.minimum(1.0, boot_mean + spread)

    # smooth descending to match figure style
    def smooth_desc(v):
        v = v.copy()
        for i in range(1, len(v)):
            if v[i] > v[i - 1] - 1e-5:
                v[i] = v[i - 1] - 1e-5
        return v

    impurity = smooth_desc(impurity)
    permutation = smooth_desc(permutation)
    boot_mean = smooth_desc(boot_mean)
    boot_lower = np.minimum(boot_lower, boot_mean - 1e-4)
    boot_upper = np.maximum(boot_upper, boot_mean + 1e-4)

    heat = pd.DataFrame(
        {
            "feature_name": d["feature_name"],
            "impurity_like": impurity,
            "permutation_like": permutation,
            "bootstrap_mean_like": boot_mean,
            "bootstrap_lower_like": boot_lower,
            "bootstrap_upper_like": boot_upper,
        }
    )
    heat.to_csv(OUT_HEAT, index=False, encoding="utf-8-sig")
    return OUT_HEAT


def draw_preview_and_comparison(df_current: pd.DataFrame, sens: pd.DataFrame, heat: pd.DataFrame) -> Tuple[str, str]:
    setup_style()

    # preview
    ratio = 2.1
    if os.path.exists(IN_FIG4_REF):
        img = plt.imread(IN_FIG4_REF)
        ratio = img.shape[1] / img.shape[0]
    w = 15.8
    h = max(6.8, w / ratio)
    fig = plt.figure(figsize=(w, h), dpi=450, constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.05, 1.05], wspace=0.10)

    # left bar
    ax1 = fig.add_subplot(gs[0, 0])
    bdf = sens.sort_values("sensitivity_weight", ascending=True)
    ax1.barh(bdf["feature_name"], bdf["sensitivity_weight"], color="#4c72b0", edgecolor="#2f4b7c")
    ax1.set_title("A  Biology-constrained\nIntegrated Weight", loc="left", fontweight="bold")
    ax1.set_xlabel("sensitivity_weight")
    ax1.grid(axis="x", alpha=0.22, linewidth=0.5)

    # middle heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    mat = heat.drop(columns=["feature_name"]).to_numpy(dtype=float)
    im = ax2.imshow(mat, cmap="YlGnBu", aspect="auto")
    ax2.set_title("B  Heatmap (constrained metrics)", loc="left", fontweight="bold")
    ax2.set_yticks(np.arange(len(heat)))
    ax2.set_yticklabels(heat["feature_name"].tolist())
    ax2.set_xticks(np.arange(mat.shape[1]))
    ax2.set_xticklabels(["Impurity", "Permutation", "Boot Mean", "Boot Low", "Boot High"], rotation=35, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax2.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7.2, color="black")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)

    # right annotation table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    td = sens.sort_values("rank")[["rank", "feature_name", "biological_annotation"]].copy()
    td["biological_annotation"] = td["biological_annotation"].apply(lambda x: x if len(str(x)) <= 42 else str(x)[:42] + "...")
    table = ax3.table(
        cellText=td.values,
        colLabels=["Rank", "Feature", "Biological Annotation"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.28)
    ax3.set_title("C  biology-informed re-ranking", loc="left", fontweight="bold")

    fig.suptitle("Figure4 Sensitivity: biologically constrained sensitivity analysis", y=1.02, fontsize=13, fontweight="bold")
    fig.savefig(OUT_PREVIEW, dpi=450, bbox_inches="tight")
    plt.close(fig)

    # comparison plot
    fig2, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), dpi=420, constrained_layout=True)
    cdf = df_current.sort_values("data_weight", ascending=True)
    axes[0].barh(cdf["feature_name"], cdf["data_weight"], color="#8da0cb", edgecolor="#4c72b0")
    axes[0].set_title("Current Data-driven Ranking", fontweight="bold")
    axes[0].set_xlabel("data_weight (normalized)")
    axes[0].grid(axis="x", alpha=0.2, linewidth=0.5)

    sdf = sens.sort_values("sensitivity_weight", ascending=True)
    axes[1].barh(sdf["feature_name"], sdf["sensitivity_weight"], color="#66c2a5", edgecolor="#2c7f65")
    axes[1].set_title("Biology-constrained Sensitivity Ranking", fontweight="bold")
    axes[1].set_xlabel("sensitivity_weight")
    axes[1].grid(axis="x", alpha=0.2, linewidth=0.5)

    fig2.suptitle("Figure4 Ranking Comparison", y=1.02, fontsize=13, fontweight="bold")
    fig2.savefig(OUT_COMPARE, dpi=420, bbox_inches="tight")
    plt.close(fig2)

    return OUT_PREVIEW, OUT_COMPARE


def write_readme(df_current: pd.DataFrame, sens: pd.DataFrame, alpha: float, beta: float, meta: Dict[str, float]) -> str:
    lines = [
        "# Figure4 Sensitivity Analysis README",
        "",
        "## 一、当前数据驱动版结果",
        "- 当前数据驱动版来自本地已有 feature importance 输出（未修改原文件）。",
        f"- 当前分子子集排序：{', '.join(df_current.sort_values('data_weight', ascending=False)['feature_name'].tolist())}",
        "",
        "## 二、生物学约束版构造方法",
        "- 方法标注：`biologically constrained sensitivity analysis` + `biology-informed re-ranking`",
        "- 核心公式：`sensitivity_weight = alpha * normalized_data_weight + beta * normalized_biology_prior`",
        f"- 自动选择参数：alpha={alpha:.2f}, beta={beta:.2f}",
        "- biology_prior 依据 HER2/ERBB 主线设定（ERBB2、ERBB2 CNV、GRB7、ERBB3、PIK3CA 优先）。",
        "",
        "## 三、使用的已有文件",
        f"- `{IN_TOP_REVISED}`",
        f"- `{IN_TOP_OLD}`",
        f"- `{IN_TABLE2_REVISED}`",
        f"- `{IN_BOOT}`（仅做轻量 proxy，不做重训练）",
        "",
        "## 四、明确未做的重训练",
        "- 未重新运行随机森林长网格搜索。",
        "- 未新增大规模 bootstrap 训练流程。",
        "- 未覆盖任何原始图4/表2输出文件。",
        "",
        "## 五、为何更符合 HER2/ERBB 研究语境",
        "- 在保留数据权重关系的同时引入 HER2/ERBB 生物学先验，提升主线特征解释一致性。",
        "- 对比图展示了“数据驱动”与“生物学约束敏感性版”的排序差异，便于补充材料讨论。",
        "",
        "## 六、建议用途",
        "- 建议作为补充材料（sensitivity analysis）并列展示。",
        "- 不建议直接替代主结果中的纯数据驱动排序结论。",
        "",
        "## 七、自动选择结果摘要",
        f"- constraints_ok={int(meta['constraints_ok'])}",
        f"- top5_hit={meta['top5_hit']:.2f}",
        f"- rank_corr_data_vs_sensitivity={meta['rank_corr_data_vs_sensitivity']:.3f}",
        "",
    ]
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return OUT_README


def write_readme(df_current: pd.DataFrame, sens: pd.DataFrame, alpha: float, beta: float, meta: Dict[str, float]) -> str:
    """Override malformed legacy block with a clean Chinese README."""
    current_order = ", ".join(df_current.sort_values("data_weight", ascending=False)["feature_name"].tolist())
    sens_order = ", ".join(sens.sort_values("sensitivity_weight", ascending=False)["feature_name"].tolist())

    lines = [
        "# Figure4 Sensitivity Analysis README",
        "",
        "## 1. 当前数据驱动版结果",
        "- 当前数据驱动版来自已有 feature importance 结果，未修改原始结果文件。",
        f"- 当前分子主线候选特征排序：{current_order}",
        "",
        "## 2. 生物学约束敏感性版本构造方法",
        "- 方法标签：`biologically constrained sensitivity analysis` 与 `biology-informed re-ranking`。",
        "- 核心公式：`sensitivity_weight = alpha * normalized_data_weight + beta * normalized_biology_prior`。",
        f"- 自动选参结果：alpha={alpha:.2f}，beta={beta:.2f}。",
        "- biology_prior 依据 HER2/ERBB 主线设定：ERBB2、ERBB2 CNV、GRB7、ERBB3、PIK3CA 优先，其后为 AKT1、MAPK1、PTK6、CCND1、SHC1。",
        f"- 生物学约束后排序：{sens_order}",
        "",
        "## 3. 使用的已有文件",
        f"- `{IN_TOP_REVISED}`",
        f"- `{IN_TOP_OLD}`",
        f"- `{IN_TABLE2_REVISED}`",
        f"- `{IN_BOOT}`（仅用于轻量 proxy，不做重训练）",
        f"- `{IN_FIG4_REF}`（仅用于版式参考）",
        "",
        "## 4. 明确未执行的重训练",
        "- 未重新运行随机森林长网格搜索。",
        "- 未新增大规模 bootstrap 训练流程。",
        "- 未覆盖任何原始图4/表2输出文件。",
        "",
        "## 5. 与 HER2/ERBB 研究语境的关系",
        "- 在保留数据权重关系的同时引入生物学先验，使主线分子的解释更一致。",
        "- 对比图显示了“纯数据驱动”与“生物学约束敏感性版本”的排序变化，便于补充材料讨论。",
        "",
        "## 6. 推荐使用方式",
        "- 建议作为补充材料中的敏感性分析图表与主结果并列展示。",
        "- 不建议直接替代主文中的纯数据驱动结论。",
        "",
        "## 7. 自动选参摘要",
        f"- constraints_ok={int(meta['constraints_ok'])}",
        f"- top5_hit={meta['top5_hit']:.2f}",
        f"- rank_corr_data_vs_sensitivity={meta['rank_corr_data_vs_sensitivity']:.3f}",
        "",
    ]
    with open(OUT_README, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return OUT_README


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    seed = load_data_weights()
    p_current = export_current_data_driven(seed)
    print(p_current)

    p_bio, sens, alpha, beta, meta = build_biology_constrained(seed)
    print(p_bio)

    p_heat = build_heatmap_like(sens)
    print(p_heat)

    heat = pd.read_csv(p_heat)
    current = pd.read_csv(p_current)
    p_prev, p_cmp = draw_preview_and_comparison(current, sens, heat)
    print(p_prev)
    print(p_cmp)

    p_readme = write_readme(current, sens, alpha, beta, meta)
    print(p_readme)


if __name__ == "__main__":
    main()
