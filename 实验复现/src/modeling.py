#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR


@dataclass
class ModelBundle:
    name: str
    model: object
    y_pred: np.ndarray
    params: Dict


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: Dict
) -> Tuple[RandomForestRegressor, Dict]:
    mode = cfg["modeling"]["mode"]
    if mode == "full":
        grid = cfg["modeling"]["rf_grid_full"]
    elif mode == "quick":
        grid = cfg["modeling"]["rf_grid_quick"]
    else:
        grid = cfg["modeling"]["rf_grid_lite"]

    base = RandomForestRegressor(
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=int(cfg["modeling"].get("n_jobs", -1)),
    )
    cv = KFold(
        n_splits=int(cfg["modeling"]["cv_folds"]),
        shuffle=True,
        random_state=int(cfg["project"]["random_seed"]),
    )
    gs = GridSearchCV(
        estimator=base,
        param_grid=grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=int(cfg["modeling"].get("n_jobs", -1)),
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_svr(X_train: pd.DataFrame, y_train: pd.Series, cfg: Dict) -> Tuple[SVR, Dict]:
    param_grid = cfg["modeling"]["svr_grid"]
    cv = KFold(
        n_splits=int(cfg["modeling"]["cv_folds"]),
        shuffle=True,
        random_state=int(cfg["project"]["random_seed"]),
    )
    gs = GridSearchCV(
        estimator=SVR(kernel="rbf"),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=int(cfg["modeling"].get("n_jobs", -1)),
        verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_params_
