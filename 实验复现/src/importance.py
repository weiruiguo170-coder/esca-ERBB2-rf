from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_feature_importance(
    rf_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    impurity_series = pd.Series(
        rf_model.feature_importances_,
        index=rf_model.feature_names_in_,
        name="impurity_importance",
    )

    perm_result = permutation_importance(
        rf_model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="r2",
        n_jobs=-1,
    )
    permutation_series = pd.Series(
        perm_result.importances_mean,
        index=X_test.columns,
        name="permutation_importance",
    )

    importance = pd.concat([impurity_series, permutation_series], axis=1).fillna(0.0)
    importance = importance.reset_index().rename(columns={"index": "feature"})

    importance["rank_impurity"] = importance["impurity_importance"].rank(
        ascending=False, method="average"
    )
    importance["rank_permutation"] = importance["permutation_importance"].rank(
        ascending=False, method="average"
    )
    importance["combined_rank"] = (
        importance["rank_impurity"] + importance["rank_permutation"]
    ) / 2.0

    return importance.sort_values("combined_rank", ascending=True)


def build_spearman_clusters(
    X_train: pd.DataFrame,
    feature_list: list[str],
    threshold: float,
) -> pd.DataFrame:
    if not feature_list:
        return pd.DataFrame(columns=["feature", "cluster_id"])

    feature_data = X_train[feature_list].copy()
    corr = feature_data.corr(method="spearman").abs().fillna(0.0)
    corr_values = corr.to_numpy()
    np.fill_diagonal(corr_values, 0.0)
    adjacency = corr_values > threshold

    features = list(corr.columns)
    visited = np.zeros(len(features), dtype=bool)
    cluster_ids = np.full(len(features), -1, dtype=int)
    cluster_id = 0

    for i in range(len(features)):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        cluster_ids[i] = cluster_id

        while stack:
            node = stack.pop()
            neighbors = np.where(adjacency[node])[0]
            for nb in neighbors:
                if not visited[nb]:
                    visited[nb] = True
                    cluster_ids[nb] = cluster_id
                    stack.append(nb)

        cluster_id += 1

    return pd.DataFrame({"feature": features, "cluster_id": cluster_ids})


def select_cluster_representatives(
    importance_table: pd.DataFrame,
    cluster_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = importance_table.merge(cluster_table, on="feature", how="left")

    if merged["cluster_id"].isna().any():
        max_cluster = int(merged["cluster_id"].dropna().max()) if merged["cluster_id"].notna().any() else -1
        for idx in merged[merged["cluster_id"].isna()].index:
            max_cluster += 1
            merged.at[idx, "cluster_id"] = max_cluster

    merged["cluster_id"] = merged["cluster_id"].astype(int)

    representatives = (
        merged.sort_values(["cluster_id", "combined_rank"], ascending=[True, True])
        .groupby("cluster_id", as_index=False)
        .first()
        .sort_values("combined_rank", ascending=True)
    )
    return merged, representatives


def feature_to_gene_symbol(feature_name: str) -> str:
    return feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
