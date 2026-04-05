#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def _set_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"


def _find_first_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for c in df.columns:
        low = c.lower()
        if any(k.lower() in low for k in keywords):
            return c
    return None


def plot_figure1_triptych(merged_df: pd.DataFrame, output_path: Path) -> Path:
    _set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    erbb2_col = _find_first_col(merged_df, ["exp_erbb2", "erbb2"])
    if erbb2_col is None:
        raise KeyError("未找到 ERBB2 表达列，无法生成图1。")
    y_col = "LN_IC50"
    if y_col not in merged_df.columns:
        raise KeyError("未找到 LN_IC50 列，无法生成图1。")

    drug_cols = [c for c in merged_df.columns if c.startswith("DRUG_")]
    if len(drug_cols) < 2:
        raise ValueError("药物 one-hot 列不足 2 个，无法生成图1三联图。")

    d1, d2 = drug_cols[:2]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), dpi=300, constrained_layout=True)

    for i, dcol in enumerate([d1, d2]):
        sub = merged_df[merged_df[dcol] > 0].copy()
        x = pd.to_numeric(sub[erbb2_col], errors="coerce")
        y = pd.to_numeric(sub[y_col], errors="coerce")
        mask = x.notna() & y.notna()
        x, y = x[mask], y[mask]
        axes[i].scatter(x, y, s=22, alpha=0.75, edgecolor="none")
        if len(x) >= 3:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            axes[i].plot(xx, coef[0] * xx + coef[1], color="#D55E00", lw=1.6)
            r, p = pearsonr(x, y)
            axes[i].text(
                0.03,
                0.95,
                f"n={len(x)}\nr={r:.3f}\np={p:.3g}",
                transform=axes[i].transAxes,
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666", alpha=0.9),
            )
        axes[i].set_title(f"{'A' if i == 0 else 'B'}  {dcol.replace('DRUG_', '')}")
        axes[i].set_xlabel("ERBB2 expression (log2(RPKM+1))")
        axes[i].set_ylabel("Drug sensitivity (LN_IC50)")

    x_all = pd.to_numeric(merged_df[erbb2_col], errors="coerce").dropna()
    axes[2].hist(x_all, bins=24, color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[2].axvline(x_all.median(), color="#D55E00", ls="--", lw=1.5, label=f"median={x_all.median():.2f}")
    axes[2].set_title("C  ERBB2 expression distribution")
    axes[2].set_xlabel("ERBB2 expression (log2(RPKM+1))")
    axes[2].set_ylabel("Count")
    axes[2].legend(frameon=False)

    fig.suptitle("Figure 1. ERBB2 expression and HER2/ERBB drug sensitivity", fontsize=12.5, y=1.03)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure3_prediction(pred_df: pd.DataFrame, output_path: Path) -> Path:
    _set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    obs = pred_df["observed_log2_ic50"].to_numpy(dtype=float)
    pred = pred_df["predicted_log2_ic50"].to_numpy(dtype=float)
    resid = pred_df["residual"].to_numpy(dtype=float)

    r2 = 1 - np.sum((obs - pred) ** 2) / np.sum((obs - np.mean(obs)) ** 2)
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=300, constrained_layout=True)

    axes[0].scatter(obs, pred, s=24, alpha=0.75, edgecolor="none")
    lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel("Observed log2(IC50)")
    axes[0].set_ylabel("Predicted log2(IC50)")
    axes[0].set_title("A  Observed vs Predicted")
    axes[0].text(
        0.04,
        0.95,
        f"n={len(obs)}\nR2={r2:.3f}\nRMSE={rmse:.3f}",
        transform=axes[0].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#666"),
        fontsize=9,
    )

    axes[1].scatter(pred, resid, s=24, alpha=0.75, edgecolor="none", color="#4C72B0")
    axes[1].axhline(0, color="red", ls="--", lw=1.2)
    axes[1].set_xlabel("Predicted log2(IC50)")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("B  Residual vs Predicted")

    fig.suptitle("Figure 3. Prediction performance and residual diagnostics", fontsize=12.5, y=1.03)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_figure4_importance(top10_df: pd.DataFrame, output_path: Path) -> Path:
    _set_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work = top10_df.sort_values("importance", ascending=True).copy()

    fig = plt.figure(figsize=(13.5, 5.8), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.barh(work["feature_name"], work["importance"], color="#4C72B0", alpha=0.9)
    ax1.set_xlabel("Feature importance")
    ax1.set_title("A  Top10 feature importance")
    ax1.grid(axis="x", alpha=0.2)

    ax2.axis("off")
    tb = work.sort_values("importance", ascending=False)[
        ["rank", "feature_name", "data_type", "biological_annotation"]
    ].copy()
    tb["biological_annotation"] = tb["biological_annotation"].astype(str).str.slice(0, 36)
    table = ax2.table(
        cellText=tb.values,
        colLabels=["Rank", "Feature", "Type", "Annotation"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.25)
    ax2.set_title("B  Top10 annotation")

    fig.suptitle("Figure 4. Key predictive features for HER2/ERBB sensitivity", fontsize=12.5, y=1.03)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return output_path
