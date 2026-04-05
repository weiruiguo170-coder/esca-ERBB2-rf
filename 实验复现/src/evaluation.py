#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "R2": r2}


def build_model_comparison_rows(
    y_test: np.ndarray,
    predictions: Dict[str, np.ndarray],
    cfg: Dict,
    feature_count: int,
    sample_count: int,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for model_name, y_pred in predictions.items():
        m = compute_regression_metrics(y_test, y_pred)
        rows.append(
            {
                "模型": model_name,
                "目标变量": cfg["feature_engineering"]["target_column"],
                "RMSE": round(m["RMSE"], 6),
                "R2": round(m["R2"], 6),
                "训练集占比": cfg["modeling"]["train_ratio"],
                "测试集占比": 1 - float(cfg["modeling"]["train_ratio"]),
                "交叉验证折数": int(cfg["modeling"]["cv_folds"]),
                "随机种子": int(cfg["project"]["random_seed"]),
                "特征规模": int(feature_count),
                "样本量": int(sample_count),
                "备注": f"mode={cfg['modeling']['mode']}",
            }
        )
    return pd.DataFrame(rows)


def extract_top10_features_from_rf(
    feature_names: List[str], importances: np.ndarray, annotations: Dict[str, str]
) -> pd.DataFrame:
    df = pd.DataFrame({"feature_name": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["data_type"] = np.where(df["feature_name"].str.contains("CNV", case=False), "拷贝数", "基因表达/其他")
    df["biological_annotation"] = df["feature_name"].map(annotations).fillna("与 HER2/ERBB 轴相关或模型筛选特征")
    return df[["rank", "feature_name", "data_type", "biological_annotation", "importance"]]


def save_core_outputs(
    out_results: Path,
    model_comparison: pd.DataFrame,
    top10_df: pd.DataFrame,
    y_test: np.ndarray,
    rf_pred: np.ndarray,
) -> Dict[str, Path]:
    out_results.mkdir(parents=True, exist_ok=True)

    p_table1 = out_results / "table1_model_performance.csv"
    p_table2 = out_results / "table2_top10_features.csv"
    p_metrics = out_results / "metrics_summary.csv"
    p_preds = out_results / "figure3_predictions.csv"
    p_model_cmp = out_results / "model_comparison.csv"

    model_comparison.to_csv(p_table1, index=False, encoding="utf-8-sig")
    top10_df.to_csv(p_table2, index=False, encoding="utf-8-sig")
    model_comparison.to_csv(p_model_cmp, index=False, encoding="utf-8-sig")

    best_row = model_comparison.sort_values("R2", ascending=False).iloc[0].to_dict()
    with p_metrics.open("w", encoding="utf-8") as f:
        json.dump(best_row, f, ensure_ascii=False, indent=2)

    preds_df = pd.DataFrame(
        {
            "observed_log2_ic50": y_test,
            "predicted_log2_ic50": rf_pred,
            "residual": y_test - rf_pred,
        }
    )
    preds_df.to_csv(p_preds, index=False, encoding="utf-8-sig")
    return {
        "table1": p_table1,
        "table2": p_table2,
        "metrics": p_metrics,
        "predictions": p_preds,
        "model_comparison": p_model_cmp,
    }
