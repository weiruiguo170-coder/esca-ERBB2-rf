from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVR


def _rf_grid(rf_cfg: dict[str, Any]) -> list[dict[str, int]]:
    n_estimators = list(
        range(
            int(rf_cfg["n_estimators_start"]),
            int(rf_cfg["n_estimators_end"]) + 1,
            int(rf_cfg["n_estimators_step"]),
        )
    )
    max_depth_values = list(
        range(int(rf_cfg["max_depth_start"]), int(rf_cfg["max_depth_end"]) + 1)
    )
    min_split_values = [int(v) for v in rf_cfg["min_samples_split_values"]]

    return [
        {
            "n_estimators": n_est,
            "max_depth": depth,
            "min_samples_split": min_split,
        }
        for n_est, depth, min_split in product(n_estimators, max_depth_values, min_split_values)
    ]


def search_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modeling_cfg: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, int], pd.DataFrame]:
    cv_folds = int(modeling_cfg["cv_folds"])
    rf_cfg = modeling_cfg["rf"]
    candidates = _rf_grid(rf_cfg)

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rows: list[dict[str, float | int]] = []

    best_score = -np.inf
    best_params = None

    for params in candidates:
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            bootstrap=True,
            oob_score=bool(rf_cfg.get("oob_score", True)),
            random_state=random_state,
            n_jobs=-1,
        )

        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))

        model.fit(X_train, y_train)
        oob_r2 = float(model.oob_score_) if hasattr(model, "oob_score_") else np.nan

        combined_score = float(np.nanmean([cv_mean, oob_r2]))
        rows.append(
            {
                "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"],
                "min_samples_split": params["min_samples_split"],
                "cv_r2_mean": cv_mean,
                "oob_r2": oob_r2,
                "combined_score": combined_score,
            }
        )

        if combined_score > best_score:
            best_score = combined_score
            best_params = params

    if best_params is None:
        raise RuntimeError("RF search failed to produce a valid parameter set.")

    search_df = pd.DataFrame(rows).sort_values("combined_score", ascending=False)
    return best_params, search_df


def fit_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, int],
    random_state: int,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        bootstrap=True,
        oob_score=True,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def search_svr(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modeling_cfg: dict[str, Any],
    random_state: int,
) -> tuple[dict[str, float], pd.DataFrame]:
    cv_folds = int(modeling_cfg["cv_folds"])
    c_values = [float(v) for v in modeling_cfg["svr"]["c_values"]]
    gamma_values = [float(v) for v in modeling_cfg["svr"]["gamma_values"]]

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_score = -np.inf
    best_params: dict[str, float] | None = None
    rows: list[dict[str, float]] = []

    for c_val, gamma_val in product(c_values, gamma_values):
        model = SVR(kernel="rbf", C=c_val, gamma=gamma_val)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring="r2", n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))

        rows.append({"C": c_val, "gamma": gamma_val, "cv_r2_mean": cv_mean})

        if cv_mean > best_score:
            best_score = cv_mean
            best_params = {"C": c_val, "gamma": gamma_val}

    if best_params is None:
        raise RuntimeError("SVR search failed to produce a valid parameter set.")

    search_df = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False)
    return best_params, search_df


def fit_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    svr_params: dict[str, float],
) -> dict[str, Any]:
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    svr_model = SVR(kernel="rbf", C=svr_params["C"], gamma=svr_params["gamma"])
    svr_model.fit(X_train, y_train)

    return {"lr": lr_model, "svr": svr_model}


def make_prediction_table(
    y_true: pd.Series,
    rf_model: RandomForestRegressor,
    baseline_models: dict[str, Any],
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    table = pd.DataFrame(index=X_test.index)
    table["y_true"] = y_true.loc[X_test.index]
    table["rf_pred"] = rf_model.predict(X_test)
    table["lr_pred"] = baseline_models["lr"].predict(X_test)
    table["svr_pred"] = baseline_models["svr"].predict(X_test)
    table.index.name = "cell_line_id"
    return table.reset_index()
