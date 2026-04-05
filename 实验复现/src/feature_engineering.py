#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureArtifacts:
    imputer: SimpleImputer
    scaler: StandardScaler
    selector: VarianceThreshold
    selected_features: List[str]


def _numeric_feature_columns(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> List[str]:
    cols = []
    for c in df.columns:
        if c == target_col or c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def build_feature_matrix(
    merged_df: pd.DataFrame, cfg: Dict
) -> tuple[pd.DataFrame, pd.Series, FeatureArtifacts]:
    target_col = cfg["feature_engineering"]["target_column"]
    drop_cols = cfg["feature_engineering"].get("drop_columns", [])
    variance_threshold = float(cfg["feature_engineering"].get("variance_threshold", 0.01))

    if target_col not in merged_df.columns:
        raise KeyError(f"目标列不存在: {target_col}")

    work = merged_df.copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work[work[target_col].notna()].copy()

    feature_cols = _numeric_feature_columns(work, target_col, drop_cols)
    X = work[feature_cols].copy()
    y = work[target_col].astype(float).copy()

    imputer = SimpleImputer(strategy="mean")
    X_imp = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    selector = VarianceThreshold(threshold=variance_threshold)
    X_sel = selector.fit_transform(X_scaled)
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]

    X_out = pd.DataFrame(X_sel, columns=selected_features, index=work.index)
    artifacts = FeatureArtifacts(
        imputer=imputer,
        scaler=scaler,
        selector=selector,
        selected_features=selected_features,
    )
    return X_out, y, artifacts


def bootstrap_expand(
    X: pd.DataFrame, y: pd.Series, target_n: int, random_seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if target_n <= len(X):
        return X.copy(), y.copy()
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(np.arange(len(X)), size=target_n, replace=True)
    Xb = X.iloc[idx].reset_index(drop=True)
    yb = y.iloc[idx].reset_index(drop=True)
    return Xb, yb


def make_train_test_split(
    X: pd.DataFrame, y: pd.Series, cfg: Dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        train_size=float(cfg["modeling"]["train_ratio"]),
        random_state=int(cfg["project"]["random_seed"]),
    )
