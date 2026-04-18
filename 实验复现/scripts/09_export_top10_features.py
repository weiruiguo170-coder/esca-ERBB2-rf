from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.importance import feature_to_gene_symbol  # noqa: E402
from src.utils import ensure_dir, load_config, project_root_from_config, read_lines, resolve_path, write_lines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export top-10 key features and background genes.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    root = project_root_from_config(args.config)
    paths = config["paths"]

    representatives = pd.read_csv(resolve_path(root, paths["representative_features"]))
    representatives = representatives.sort_values("combined_rank", ascending=True)

    top_k = int(config["importance"]["top_k"])
    top_features = representatives.head(top_k).copy()
    top_features["gene_symbol"] = top_features["feature"].map(feature_to_gene_symbol)

    top_genes = top_features["gene_symbol"].dropna().astype(str).drop_duplicates().tolist()

    background_genes = sorted(
        set(read_lines(resolve_path(root, paths["selected_expression_genes"])))
    )

    ensure_dir(resolve_path(root, paths["top_features_dir"]))
    top_features.to_csv(resolve_path(root, paths["top10_features"]), index=False)
    write_lines(resolve_path(root, paths["top10_genes"]), top_genes)
    write_lines(resolve_path(root, paths["background_genes"]), background_genes)

    print("[09] Top features and enrichment inputs exported.")
    print(f"[09] Top genes: {len(top_genes)}, background genes: {len(background_genes)}")


if __name__ == "__main__":
    main()
