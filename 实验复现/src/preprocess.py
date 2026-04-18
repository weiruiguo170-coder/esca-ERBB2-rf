from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_joint_target(drug_response: pd.DataFrame) -> pd.Series:
    ic50_cols = ["trastuzumab_ic50", "lapatinib_ic50"]
    ic50 = drug_response[["cell_line_id", *ic50_cols]].copy()

    for col in ic50_cols:
        ic50[col] = pd.to_numeric(ic50[col], errors="coerce")
        ic50.loc[ic50[col] <= 0, col] = np.nan

    ic50["trastuzumab_log2_ic50"] = np.log2(ic50["trastuzumab_ic50"])
    ic50["lapatinib_log2_ic50"] = np.log2(ic50["lapatinib_ic50"])
    ic50["joint_log2_ic50_mean"] = ic50[
        ["trastuzumab_log2_ic50", "lapatinib_log2_ic50"]
    ].mean(axis=1)

    ic50 = ic50.dropna(subset=["joint_log2_ic50_mean"])
    return ic50.set_index("cell_line_id")["joint_log2_ic50_mean"]


def build_feature_matrix(
    expression: pd.DataFrame,
    cnv: pd.DataFrame,
    pathway_genes: list[str],
    expression_prefix: str,
    cnv_prefix: str,
) -> tuple[pd.DataFrame, list[str]]:
    available_pathway_expression_genes = [gene for gene in pathway_genes if gene in expression.index]
    if not available_pathway_expression_genes:
        raise ValueError("None of the pathway genes were found in the expression matrix.")

    expression_features = expression.T.copy()
    expression_features.columns = [f"{expression_prefix}{gene}" for gene in expression_features.columns]

    available_pathway_genes = [gene for gene in pathway_genes if gene in cnv.index]
    if not available_pathway_genes:
        raise ValueError("None of the pathway genes were found in the CNV matrix.")

    cnv_subset = cnv.loc[available_pathway_genes].T.copy()
    cnv_subset.columns = [f"{cnv_prefix}{gene}" for gene in cnv_subset.columns]

    features = expression_features.join(cnv_subset, how="inner")
    if features.empty:
        raise ValueError("Feature matrix is empty after joining expression and CNV features.")

    return features, available_pathway_genes


def split_and_preprocess(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    variance_threshold: float,
    expression_prefix: str,
) -> dict[str, object]:
    aligned_ids = X.index.intersection(y.index)
    X = X.loc[aligned_ids].copy()
    y = y.loc[aligned_ids].copy()

    train_ids, test_ids = train_test_split(
        X.index,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    X_train = X.loc[train_ids].copy()
    X_test = X.loc[test_ids].copy()
    y_train = y.loc[train_ids].copy()
    y_test = y.loc[test_ids].copy()

    impute_means = X_train.mean(axis=0)
    X_train_imputed = X_train.fillna(impute_means)
    X_test_imputed = X_test.fillna(impute_means)

    scale_means = X_train_imputed.mean(axis=0)
    scale_stds = X_train_imputed.std(axis=0, ddof=0).replace(0, 1.0)
    X_train_scaled = (X_train_imputed - scale_means) / scale_stds
    X_test_scaled = (X_test_imputed - scale_means) / scale_stds

    expression_cols = [c for c in X_train_scaled.columns if c.startswith(expression_prefix)]
    non_expression_cols = [c for c in X_train_scaled.columns if c not in expression_cols]

    expression_variances = X_train_scaled[expression_cols].var(axis=0, ddof=0)
    selected_expression_cols = expression_variances[
        expression_variances >= variance_threshold
    ].index.tolist()

    selected_features = selected_expression_cols + non_expression_cols
    X_train_final = X_train_scaled[selected_features].copy()
    X_test_final = X_test_scaled[selected_features].copy()

    return {
        "X_train": X_train_final,
        "X_test": X_test_final,
        "y_train": y_train,
        "y_test": y_test,
        "selected_features": selected_features,
        "selected_expression_cols": selected_expression_cols,
        "train_ids": list(train_ids),
        "test_ids": list(test_ids),
    }


def bootstrap_training_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_samples: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(random_state)
    indices = rng.choice(np.arange(len(X_train)), size=n_samples, replace=True)

    X_bootstrap = X_train.iloc[indices].copy()
    y_bootstrap = y_train.iloc[indices].copy()

    bootstrap_source_ids = pd.Series(
        X_train.index[indices].astype(str), name="sampled_cell_line_id"
    )
    X_bootstrap.index = [f"bootstrap_{i:04d}" for i in range(len(X_bootstrap))]
    y_bootstrap.index = X_bootstrap.index

    return X_bootstrap, y_bootstrap, bootstrap_source_ids
