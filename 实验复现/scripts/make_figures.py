#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plotting import plot_figure1_triptych, plot_figure3_prediction, plot_figure4_importance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="从现有中间结果生成图1/图3/图4")
    parser.add_argument("--intermediate-dir", default="intermediate")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    intermediate_dir = Path(args.intermediate_dir)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged = intermediate_dir / "merged_dataset.csv"
    preds = results_dir / "figure3_predictions.csv"
    top10 = results_dir / "table2_top10_features.csv"

    if merged.exists():
        merged_df = pd.read_csv(merged)
        out1 = plot_figure1_triptych(merged_df, figures_dir / "figure1_main.png")
        print(f"[OK] {out1}")
    else:
        print(f"[WARN] 缺少 {merged}")

    if preds.exists():
        pred_df = pd.read_csv(preds)
        out3 = plot_figure3_prediction(pred_df, figures_dir / "figure3_prediction_performance.png")
        print(f"[OK] {out3}")
    else:
        print(f"[WARN] 缺少 {preds}")

    if top10.exists():
        top_df = pd.read_csv(top10)
        # Handle Chinese schema or generic schema.
        rename_map = {}
        if "特征名称" in top_df.columns:
            rename_map["特征名称"] = "feature_name"
        if "重要性权重" in top_df.columns:
            rename_map["重要性权重"] = "importance"
        if "排名" in top_df.columns:
            rename_map["排名"] = "rank"
        if "生物学注释" in top_df.columns:
            rename_map["生物学注释"] = "biological_annotation"
        if "数据类型" in top_df.columns:
            rename_map["数据类型"] = "data_type"
        top_df = top_df.rename(columns=rename_map)
        out4 = plot_figure4_importance(top_df, figures_dir / "figure4_feature_importance.png")
        print(f"[OK] {out4}")
    else:
        print(f"[WARN] 缺少 {top10}")


if __name__ == "__main__":
    main()
