from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.importance import build_spearman_clusters, select_cluster_representatives  # noqa: E402
from src.utils import ensure_dir, load_config, project_root_from_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster features by Spearman |r| and keep one representative per cluster."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    importance = pd.read_csv(resolve_path(root, paths["feature_importance_table"]))
    X_train = pd.read_csv(resolve_path(root, paths["x_train_bootstrap"]), index_col=0)

    feature_list = importance["feature"].tolist()
    clusters = build_spearman_clusters(
        X_train=X_train,
        feature_list=feature_list,
        threshold=float(config["importance"]["spearman_abs_threshold"]),
    )

    cluster_detail, representatives = select_cluster_representatives(
        importance_table=importance,
        cluster_table=clusters,
    )

    ensure_dir(resolve_path(root, paths["clustering_dir"]))
    cluster_detail.to_csv(resolve_path(root, paths["feature_clusters"]), index=False)
    representatives.to_csv(resolve_path(root, paths["representative_features"]), index=False)

    print("[08] Correlation clustering completed.")
    print(
        f"[08] Total features: {len(feature_list)}, "
        f"clusters: {clusters['cluster_id'].nunique()}, representatives: {len(representatives)}"
    )


if __name__ == "__main__":
    main()
