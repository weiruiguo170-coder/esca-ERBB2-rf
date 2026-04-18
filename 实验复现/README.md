# ERBB2-Axis Drug Sensitivity Reproducibility Package

## Overview
This repository is a lightweight reproducibility package for the manuscript methodology.
It is designed to reproduce the modeling workflow only (not to redistribute source datasets).

Key principles:
- Method-aligned workflow only
- No raw data distribution
- No personal, institutional, or submission metadata
- Lightweight and GitHub-ready project structure

## Data Policy
This repository does **not** provide:
- Raw GDSC/CCLE downloads
- Any patient-level or sensitive data
- Full manuscript text or submission materials
- Large intermediate/model output files

Users must download public data independently from GDSC and CCLE and place prepared files under `data/input/`.

## Method Alignment (Implemented)
The code follows the manuscript workflow:
1. Public sources: GDSC + CCLE
2. Study scope: upper GI, ERBB2-axis-relevant cell lines
3. Drugs: trastuzumab and lapatinib
4. Endpoint: mean of `log2(IC50_trastuzumab)` and `log2(IC50_lapatinib)`
5. Features: transcriptome-wide mRNA expression (including ERBB2-pathway mRNA genes) + ERBB2-pathway CNV features
6. Split: train/test = 7:3, `random_state=42`
7. Missing value handling: train-mean imputation per feature
8. Standardization: train-set mean/std Z-score
9. Feature filtering: variance threshold 0.01 on expression features
10. Bootstrap: training set only, generate 480 simulated samples
11. Main model: RandomForestRegressor
12. RF search:
   - `n_estimators`: 200-800 (step 100)
   - `max_depth`: 4-14
   - `min_samples_split`: {2,4,6,8,10}
13. RF selection criterion: combined 5-fold CV R² + OOB R²
14. Baselines: Linear Regression and SVR (RBF)
15. SVR grid:
   - `C` in {0.1, 1, 10, 100}
   - `gamma` in {1e-4, 1e-3, 1e-2, 1e-1}
16. Evaluation metrics: RMSE and R²
17. Importance: impurity importance + permutation importance, then rank merge
18. Redundancy removal: Spearman clustering with `|r| > 0.7`, keep top-ranked representative per cluster
19. Enrichment: top 10 key features, KEGG + GO via R `clusterProfiler`, background = all expression genes used in the model, significance `P < 0.05` and `FDR < 0.25`

## Input Files
Place these files in `data/input/`:
- `expression.csv`
- `cnv.csv`
- `drug_response.csv`
- `erbb2_pathway_genes.txt`

Use `data/schema/` templates as format references.

### expression.csv
- Rows: genes
- Columns: `gene_symbol`, then cell-line IDs
- Values: mRNA expression values

### cnv.csv
- Rows: genes
- Columns: `gene_symbol`, then cell-line IDs
- Values: CNV values

### drug_response.csv
Required columns:
- `cell_line_id`
- `source` (must include GDSC/CCLE labels)
- `tissue_group` (use `upper_gi` for included rows)
- `erbb2_axis_relevant` (1 = include, 0 = exclude)
- `trastuzumab_ic50`
- `lapatinib_ic50`

### Alignment rules
- Cell-line IDs must match across expression/CNV/drug-response files
- Gene symbols should be standardized text labels
- IC50 values must be positive numeric values for log2 transformation

## Environment Setup
### Python
```bash
conda env create -f environment.yml
conda activate erbb2-repro
```

or

```bash
pip install -r requirements.txt
```

### R (for script 10)
Install packages before enrichment:
- `clusterProfiler`
- `org.Hs.eg.db`
- `yaml`

Example:
```r
install.packages("yaml")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install(c("clusterProfiler", "org.Hs.eg.db"))
```

## Reproducibility Run Order
Run from repository root:

```bash
python scripts/01_prepare_public_data.py --config configs/config.yaml
python scripts/02_build_target_and_features.py --config configs/config.yaml
python scripts/03_split_preprocess.py --config configs/config.yaml
python scripts/04_bootstrap_training_set.py --config configs/config.yaml
python scripts/05_train_rf_and_baselines.py --config configs/config.yaml
python scripts/06_evaluate_models.py --config configs/config.yaml
python scripts/07_feature_importance.py --config configs/config.yaml
python scripts/08_correlation_clustering.py --config configs/config.yaml
python scripts/09_export_top10_features.py --config configs/config.yaml
Rscript scripts/10_run_enrichment.R configs/config.yaml
```

## Output
All generated outputs go to `results/` during runtime and are git-ignored by default.
The repository only keeps `results/.gitkeep` as a placeholder.

## Reproducibility Notes
- Using different GDSC/CCLE versions can change selected feature counts and model metrics.
- With manuscript-consistent data versions and filtering settings, selected high-variance expression features are expected to be close to the reported scale (around 11842), but not hard-coded.
- The manuscript reports a simulation-oriented scenario; exact numbers may vary.
- Reproduced outcomes should trend toward the reported behavior (RF better RMSE/R² than LR/SVR, similar key-feature ranking tendencies, and related pathway enrichment patterns).

## License
MIT License (see `LICENSE`).
