#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="从现有结果生成表1和表2")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_cmp = results_dir / "model_comparison.csv"
    table1 = results_dir / "table1_model_performance.csv"
    table2 = results_dir / "table2_top10_features.csv"
    raw_top = results_dir / "feature_importance_raw_top.csv"

    if model_cmp.exists():
        df_cmp = pd.read_csv(model_cmp)
        df_cmp.to_csv(table1, index=False, encoding="utf-8-sig")
        print(f"[OK] table1: {table1}")
    else:
        print(f"[WARN] 未找到 {model_cmp}")

    if raw_top.exists():
        df = pd.read_csv(raw_top)
        if "rank" not in df.columns:
            df = df.sort_values(df.columns[-1], ascending=False).head(10).reset_index(drop=True)
            df.insert(0, "rank", range(1, len(df) + 1))
        df.to_csv(table2, index=False, encoding="utf-8-sig")
        print(f"[OK] table2: {table2}")
    else:
        print(f"[WARN] 未找到 {raw_top}")


if __name__ == "__main__":
    main()
