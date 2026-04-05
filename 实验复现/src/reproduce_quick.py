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
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.svm import SVR


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_BOOT = os.path.join(BASE_DIR, "intermediate", "bootstrap_dataset.csv")
INPUT_MERGED = os.path.join(BASE_DIR, "intermediate", "merged_dataset.csv")
INPUT_DIST = os.path.join(BASE_DIR, "results", "bootstrap_distribution_summary.csv")
INPUT_MATCH = os.path.join(BASE_DIR, "results", "matching_report.csv")

OUT_RESULTS = os.path.join(BASE_DIR, "results_quick")
OUT_FIGS = os.path.join(BASE_DIR, "figures_quick")
OUT_INTER = os.path.join(BASE_DIR, "intermediate_quick")
OUT_LOG = os.path.join(BASE_DIR, "logs", "quick_run.log")


def ensure_dirs() -> None:
    for d in [OUT_RESULTS, OUT_FIGS, OUT_INTER, os.path.dirname(OUT_LOG)]:
        os.makedirs(d, exist_ok=True)


def load_bootstrap() -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(INPUT_BOOT):
        raise FileNotFoundError(f"缺少输入文件: {INPUT_BOOT}")
    boot = pd.read_csv(INPUT_BOOT, low_memory=False)
    feature_cols = [
        c
        for c in boot.columns
        if c not in {"LN_IC50", "CCLE_ID", "DRUG_NAME", "CELL_LINE_NAME", "bootstrap_source_index"}
    ]
    if "LN_IC50" not in boot.columns:
        raise RuntimeError("bootstrap_dataset.csv 中缺少 LN_IC50 列。")
    return boot, feature_cols


def train_quick(boot: pd.DataFrame, feature_cols: list[str]):
    X = boot[feature_cols].to_numpy(dtype=float)
    y = boot["LN_IC50"].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestRegressor(random_state=42, n_jobs=1)
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [6, 10],
        "min_samples_split": [2, 6],
    }
    rf_search = GridSearchCV(
        rf,
        rf_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    y_pred_rf = best_rf.predict(X_test)

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)

    # Quick-mode SVR: smaller grid to avoid waiting for long_run completion
    svr = SVR(kernel="rbf")
    svr_grid = {"C": [1, 10], "gamma": [1e-3, 1e-2]}
    svr_search = GridSearchCV(
        svr,
        svr_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    svr_search.fit(X_train, y_train)
    y_pred_svr = svr_search.best_estimator_.predict(X_test)

    rf_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    rf_r2 = float(r2_score(y_test, y_pred_rf))
    lin_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_lin)))
    lin_r2 = float(r2_score(y_test, y_pred_lin))
    svr_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_svr)))
    svr_r2 = float(r2_score(y_test, y_pred_svr))

    return {
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_rf": y_pred_rf,
        "best_rf": best_rf,
        "rf_search": rf_search,
        "svr_search": svr_search,
        "metrics": {
            "rf_rmse": rf_rmse,
            "rf_r2": rf_r2,
            "rf_cv_rmse": float(-rf_search.best_score_),
            "lin_rmse": lin_rmse,
            "lin_r2": lin_r2,
            "svr_rmse": svr_rmse,
            "svr_r2": svr_r2,
            "svr_cv_rmse": float(-svr_search.best_score_),
        },
    }


def save_metrics_summary(result_obj: dict) -> str:
    m = result_obj["metrics"]
    rf_search = result_obj["rf_search"]
    rows = [
        {
            "mode": "quick",
            "model": "RandomForestRegressor",
            "test_rmse": m["rf_rmse"],
            "test_r2": m["rf_r2"],
            "cv_best_rmse": m["rf_cv_rmse"],
            "best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
            "rf_grid_note": "quick grid: n_estimators=[200,400], max_depth=[6,10], min_samples_split=[2,6]",
            "ln_ic50_note": "沿用 LN_IC50（自然对数），未换算为 log2(IC50)",
        }
    ]
    out = os.path.join(OUT_RESULTS, "metrics_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_model_comparison(result_obj: dict) -> str:
    m = result_obj["metrics"]
    rf_search = result_obj["rf_search"]
    svr_search = result_obj["svr_search"]
    rows = [
        {
            "mode": "quick",
            "model": "RandomForestRegressor",
            "test_rmse": m["rf_rmse"],
            "test_r2": m["rf_r2"],
            "cv_best_rmse": m["rf_cv_rmse"],
            "cv_best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
        },
        {
            "mode": "quick",
            "model": "LinearRegression",
            "test_rmse": m["lin_rmse"],
            "test_r2": m["lin_r2"],
            "cv_best_rmse": np.nan,
            "cv_best_params": "",
        },
        {
            "mode": "quick",
            "model": "SVR_RBF",
            "test_rmse": m["svr_rmse"],
            "test_r2": m["svr_r2"],
            "cv_best_rmse": m["svr_cv_rmse"],
            "cv_best_params": json.dumps(svr_search.best_params_, ensure_ascii=False),
        },
    ]
    out = os.path.join(OUT_RESULTS, "model_comparison.csv")
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    return out


def save_top10_and_plots(result_obj: dict, feature_cols: list[str]) -> tuple[str, str, str, str]:
    model = result_obj["best_rf"]
    X_test = result_obj["X_test"]
    y_test = result_obj["y_test"]
    y_pred = result_obj["y_pred_rf"]

    impurity = model.feature_importances_
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    perm_mean = perm.importances_mean

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "impurity_importance": impurity,
            "permutation_importance": perm_mean,
        }
    )

    def norm(s: pd.Series) -> pd.Series:
        mn = float(s.min())
        mx = float(s.max())
        if np.isclose(mn, mx):
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - mn) / (mx - mn)

    df["impurity_norm"] = norm(df["impurity_importance"])
    df["permutation_norm"] = norm(df["permutation_importance"])
    df["combined_score"] = (df["impurity_norm"] + df["permutation_norm"]) / 2.0
    top10 = df.sort_values("combined_score", ascending=False).head(10)

    out_top = os.path.join(OUT_RESULTS, "top10_features.csv")
    top10.to_csv(out_top, index=False, encoding="utf-8-sig")

    out_scatter = os.path.join(OUT_FIGS, "true_vs_pred.png")
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    lo = float(min(y_test.min(), y_pred.min()))
    hi = float(max(y_test.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], "--")
    plt.xlabel("真实 LN_IC50")
    plt.ylabel("预测 LN_IC50")
    plt.title("真实值 vs 预测值（Quick RF）")
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=300)
    plt.close()

    out_resid = os.path.join(OUT_FIGS, "residuals.png")
    resid = y_test - y_pred
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, resid, alpha=0.7)
    plt.axhline(0.0, linestyle="--")
    plt.xlabel("预测 LN_IC50")
    plt.ylabel("残差（真实-预测）")
    plt.title("残差图（Quick RF）")
    plt.tight_layout()
    plt.savefig(out_resid, dpi=300)
    plt.close()

    out_bar = os.path.join(OUT_FIGS, "top10_feature_importance.png")
    bar_df = top10.iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(bar_df["feature"], bar_df["combined_score"])
    plt.xlabel("综合重要性分数")
    plt.ylabel("特征")
    plt.title("Top10特征重要性（Quick）")
    plt.tight_layout()
    plt.savefig(out_bar, dpi=300)
    plt.close()

    return out_top, out_scatter, out_resid, out_bar


def save_context_copy() -> None:
    for src, name in [
        (INPUT_MERGED, "merged_dataset.csv"),
        (INPUT_BOOT, "bootstrap_dataset.csv"),
        (INPUT_DIST, "bootstrap_distribution_summary.csv"),
        (INPUT_MATCH, "matching_report.csv"),
    ]:
        if os.path.exists(src):
            dst = os.path.join(OUT_INTER, name)
            if not os.path.exists(dst):
                # keep quick snapshot as hard evidence of input state
                with open(src, "rb") as fr, open(dst, "wb") as fw:
                    fw.write(fr.read())


def write_log(messages: list[str]) -> None:
    with open(OUT_LOG, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(m + "\n")


def main() -> None:
    start = datetime.now()
    logs = [f"start={start.isoformat()}"]
    ensure_dirs()
    save_context_copy()

    boot, feature_cols = load_bootstrap()
    logs.append(f"bootstrap_rows={len(boot)}")
    logs.append(f"feature_cols={len(feature_cols)}")

    result_obj = train_quick(boot, feature_cols)
    m_path = save_metrics_summary(result_obj)
    c_path = save_model_comparison(result_obj)
    t_path, p1, p2, p3 = save_top10_and_plots(result_obj, feature_cols)

    end = datetime.now()
    logs.append(f"end={end.isoformat()}")
    logs.append(f"metrics_summary={m_path}")
    logs.append(f"model_comparison={c_path}")
    logs.append(f"top10={t_path}")
    logs.append(f"fig1={p1}")
    logs.append(f"fig2={p2}")
    logs.append(f"fig3={p3}")
    write_log(logs)

    print(f"[quick] metrics_summary: {m_path}")
    print(f"[quick] model_comparison: {c_path}")
    print(f"[quick] top10_features: {t_path}")
    print(f"[quick] figure: {p1}")
    print(f"[quick] figure: {p2}")
    print(f"[quick] figure: {p3}")


if __name__ == "__main__":
    main()

