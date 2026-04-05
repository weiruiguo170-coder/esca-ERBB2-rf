#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BOOT_PATH = os.path.join(BASE_DIR, "intermediate", "bootstrap_dataset.csv")
OUT_RESULTS = os.path.join(BASE_DIR, "results_quick")
OUT_FIGS = os.path.join(BASE_DIR, "figures_quick")
OUT_LOG = os.path.join(BASE_DIR, "logs", "quick_lite_run.log")


def ensure_dirs() -> None:
    os.makedirs(OUT_RESULTS, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_LOG), exist_ok=True)


def load_data() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    df = pd.read_csv(BOOT_PATH, low_memory=False)
    feature_cols = [
        c
        for c in df.columns
        if c not in {"LN_IC50", "CCLE_ID", "DRUG_NAME", "CELL_LINE_NAME", "bootstrap_source_index"}
    ]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["LN_IC50"].to_numpy(dtype=float)
    return df, feature_cols, X, y


def train_models(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_grid = {
        "n_estimators": [200, 300],
        "max_depth": [8],
        "min_samples_split": [2],
    }
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    metrics = {
        "rf_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_rf))),
        "rf_r2": float(r2_score(y_test, y_pred_rf)),
        "rf_cv_rmse": float(-rf_search.best_score_),
        "lin_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_lin))),
        "lin_r2": float(r2_score(y_test, y_pred_lin)),
        "rf_best_params": rf_search.best_params_,
    }
    return best_rf, metrics, y_test, y_pred_rf


def save_metrics(metrics: dict) -> str:
    out = os.path.join(OUT_RESULTS, "metrics_summary.csv")
    pd.DataFrame(
        [
            {
                "mode": "quick_lite",
                "model": "RandomForestRegressor",
                "test_rmse": metrics["rf_rmse"],
                "test_r2": metrics["rf_r2"],
                "cv_best_rmse": metrics["rf_cv_rmse"],
                "best_params": json.dumps(metrics["rf_best_params"], ensure_ascii=False),
                "note": "极小网格 + 3折CV；跳过SVR；跳过permutation importance",
                "ln_ic50_note": "LN_IC50按自然对数使用，未换算log2(IC50)",
            }
        ]
    ).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_model_compare(metrics: dict) -> str:
    out = os.path.join(OUT_RESULTS, "model_comparison.csv")
    pd.DataFrame(
        [
            {
                "mode": "quick_lite",
                "model": "RandomForestRegressor",
                "test_rmse": metrics["rf_rmse"],
                "test_r2": metrics["rf_r2"],
                "cv_best_rmse": metrics["rf_cv_rmse"],
                "cv_best_params": json.dumps(metrics["rf_best_params"], ensure_ascii=False),
            },
            {
                "mode": "quick_lite",
                "model": "LinearRegression",
                "test_rmse": metrics["lin_rmse"],
                "test_r2": metrics["lin_r2"],
                "cv_best_rmse": np.nan,
                "cv_best_params": "",
            },
            {
                "mode": "quick_lite",
                "model": "SVR_RBF",
                "test_rmse": np.nan,
                "test_r2": np.nan,
                "cv_best_rmse": np.nan,
                "cv_best_params": "SKIPPED_IN_QUICK_LITE",
            },
        ]
    ).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_top10_and_figs(best_rf, feature_cols: list[str], y_test: np.ndarray, y_pred: np.ndarray):
    imp = best_rf.feature_importances_
    fi = pd.DataFrame({"feature": feature_cols, "impurity_importance": imp})
    fi["combined_score"] = fi["impurity_importance"]
    top10 = fi.sort_values("combined_score", ascending=False).head(10).copy()

    top_path = os.path.join(OUT_RESULTS, "top10_features.csv")
    top10.to_csv(top_path, index=False, encoding="utf-8-sig")

    p1 = os.path.join(OUT_FIGS, "true_vs_pred.png")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("真实 LN_IC50")
    plt.ylabel("预测 LN_IC50")
    plt.title("真实值 vs 预测值（Quick Lite）")
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    p2 = os.path.join(OUT_FIGS, "residuals.png")
    resid = y_test - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, resid, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("预测 LN_IC50")
    plt.ylabel("残差（真实-预测）")
    plt.title("残差图（Quick Lite）")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()

    p3 = os.path.join(OUT_FIGS, "top10_feature_importance.png")
    top_plot = top10.iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_plot["feature"], top_plot["combined_score"])
    plt.xlabel("重要性分数（Impurity）")
    plt.ylabel("特征")
    plt.title("Top10 特征重要性（Quick Lite）")
    plt.tight_layout()
    plt.savefig(p3, dpi=300)
    plt.close()

    return top_path, p1, p2, p3


def main() -> None:
    t0 = datetime.now()
    ensure_dirs()
    _, feature_cols, X, y = load_data()
    best_rf, metrics, y_test, y_pred = train_models(X, y)

    m_path = save_metrics(metrics)
    c_path = save_model_compare(metrics)
    t_path, p1, p2, p3 = save_top10_and_figs(best_rf, feature_cols, y_test, y_pred)

    t1 = datetime.now()
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        f.write(f"start={t0.isoformat()}\n")
        f.write(f"end={t1.isoformat()}\n")
        f.write(f"metrics={m_path}\n")
        f.write(f"comparison={c_path}\n")
        f.write(f"top10={t_path}\n")
        f.write(f"fig1={p1}\n")
        f.write(f"fig2={p2}\n")
        f.write(f"fig3={p3}\n")

    print(m_path)
    print(c_path)
    print(t_path)
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()

