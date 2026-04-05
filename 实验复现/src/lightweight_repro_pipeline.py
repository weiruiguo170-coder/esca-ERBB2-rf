from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path) -> Dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def normalize_drug_name(name: str) -> str:
    n = str(name).strip().lower()
    if "trast" in n or "曲妥珠" in n:
        return "Trastuzumab"
    if "lapat" in n or "拉帕替尼" in n:
        return "Lapatinib"
    return str(name).strip()


def minmax_norm(values: pd.Series) -> pd.Series:
    arr = values.astype(float)
    v_min = float(arr.min())
    v_max = float(arr.max())
    if abs(v_max - v_min) < 1e-12:
        return pd.Series(np.ones(len(arr)), index=values.index)
    return (arr - v_min) / (v_max - v_min)


def _resolve(cfg: Dict, rel_path: str) -> Path:
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _artifact_paths(cfg: Dict) -> Dict[str, Path]:
    base = _resolve(cfg, cfg["paths"]["output_dir"])
    processed = base / "processed"
    model_dir = base / "models"
    metrics = base / "metrics"
    feat_dir = base / "feature_importance"
    figs = base / "figures"
    return {
        "base": base,
        "processed_dir": processed,
        "model_dir": model_dir,
        "metrics_dir": metrics,
        "feature_dir": feat_dir,
        "figures_dir": figs,
        "train_processed": processed / "train_processed.csv",
        "test_processed": processed / "test_processed.csv",
        "train_unscaled": processed / "train_unscaled.csv",
        "test_unscaled": processed / "test_unscaled.csv",
        "feature_columns": processed / "feature_columns.json",
        "preprocess_meta": processed / "preprocess_meta.json",
        "bootstrap_train": processed / "train_bootstrap.csv",
        "rf_model": model_dir / "rf_best.joblib",
        "rf_selection": metrics / "rf_selection_scores.csv",
        "rf_metrics": metrics / "rf_metrics.json",
        "baseline_metrics": metrics / "baseline_metrics.json",
        "predictions": metrics / "predictions_demo.csv",
        "feature_importance": feat_dir / "feature_importance_demo.csv",
        "representatives": feat_dir / "representative_features_demo.csv",
        "clusters": feat_dir / "feature_clusters_demo.csv",
        "summary": base / "run_summary.md",
        "fig_scatter": figs / "prediction_scatter_demo.png",
        "fig_importance": figs / "feature_importance_demo.png",
    }


def _ensure_dirs(paths: Dict[str, Path]) -> None:
    for key in ["base", "processed_dir", "model_dir", "metrics_dir", "feature_dir", "figures_dir"]:
        paths[key].mkdir(parents=True, exist_ok=True)


@dataclass
class StageResult:
    message: str
    files: List[str]


def generate_demo_input(cfg: Dict, force: bool = False) -> Path:
    out = _resolve(cfg, cfg["paths"]["input_csv"])
    if out.exists() and not force:
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["project"]["random_seed"])
    rng = np.random.default_rng(seed)
    features = list(cfg["data"]["required_features"])

    n_cell_lines = int(cfg["runtime"]["demo"]["n_cell_lines"])
    cell_ids = [f"ESCA_CL_{i:03d}" for i in range(1, n_cell_lines + 1)]
    tissues = ["esophagus"] * n_cell_lines

    erbb2 = rng.normal(8.0, 1.2, size=n_cell_lines)
    grb7 = erbb2 * 0.72 + rng.normal(0.0, 0.6, size=n_cell_lines)
    erbb3 = erbb2 * 0.44 + rng.normal(0.0, 0.8, size=n_cell_lines)
    pik3ca = rng.normal(5.5, 1.0, size=n_cell_lines)
    akt1 = rng.normal(6.1, 0.8, size=n_cell_lines)
    mapk1 = rng.normal(6.3, 0.9, size=n_cell_lines)
    cnd1 = rng.normal(4.9, 0.7, size=n_cell_lines)
    shc1 = rng.normal(5.7, 0.9, size=n_cell_lines)
    ptk6 = rng.normal(4.6, 0.8, size=n_cell_lines)
    cnv = np.clip(np.round((erbb2 - 6.5) + rng.normal(2.5, 0.9, size=n_cell_lines), 2), 0.0, None)

    feature_matrix = pd.DataFrame(
        {
            "EXP_ERBB2": erbb2,
            "EXP_GRB7": grb7,
            "EXP_ERBB3": erbb3,
            "EXP_PIK3CA": pik3ca,
            "EXP_AKT1": akt1,
            "EXP_MAPK1": mapk1,
            "EXP_PTK6": ptk6,
            "EXP_CCND1": cnd1,
            "EXP_SHC1": shc1,
            "CNV_ERBB2": cnv,
        }
    )
    feature_matrix = feature_matrix[features]

    rows: List[Dict] = []
    for i, cid in enumerate(cell_ids):
        base = 3.35 - 0.22 * feature_matrix.loc[i, "EXP_ERBB2"] - 0.08 * feature_matrix.loc[i, "CNV_ERBB2"]
        trast_log2 = base + rng.normal(0.0, 0.22)
        lapat_log2 = base + 0.35 + rng.normal(0.0, 0.24)
        for drug_name, log2_ic50 in [("Trastuzumab", trast_log2), ("Lapatinib", lapat_log2)]:
            record = {
                cfg["data"]["id_col"]: cid,
                cfg["data"]["tissue_col"]: tissues[i],
                cfg["data"]["drug_col"]: drug_name,
                cfg["data"]["ic50_col"]: float(2.0 ** log2_ic50),
            }
            for col in features:
                record[col] = float(feature_matrix.loc[i, col])
            rows.append(record)

    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    return out


def _validate_columns(df: pd.DataFrame, required_cols: Sequence[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")


def _prepare_endpoint_table(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, List[str]]:
    data_cfg = cfg["data"]
    id_col = data_cfg["id_col"]
    tissue_col = data_cfg["tissue_col"]
    drug_col = data_cfg["drug_col"]
    ic50_col = data_cfg["ic50_col"]
    feature_cols = list(data_cfg["required_features"])
    target_col = cfg["target"]["name"]

    _validate_columns(df, [id_col, drug_col, ic50_col] + feature_cols)

    work = df.copy()
    if tissue_col in work.columns and data_cfg.get("tissue_keywords"):
        mask = pd.Series(False, index=work.index)
        text = work[tissue_col].astype(str).str.lower()
        for kw in data_cfg["tissue_keywords"]:
            mask = mask | text.str.contains(str(kw).lower(), regex=False)
        work = work[mask].copy()

    work[drug_col] = work[drug_col].map(normalize_drug_name)
    target_drugs = [normalize_drug_name(d) for d in cfg["target"]["drugs"]]
    work = work[work[drug_col].isin(target_drugs)].copy()

    work[ic50_col] = pd.to_numeric(work[ic50_col], errors="coerce")
    work = work.dropna(subset=[ic50_col])
    work = work[work[ic50_col] > 0]
    work["log2_ic50"] = np.log2(work[ic50_col].astype(float))

    drug_wide = work.pivot_table(index=id_col, columns=drug_col, values="log2_ic50", aggfunc="mean")
    keep_drugs = [d for d in target_drugs if d in drug_wide.columns]
    if not keep_drugs:
        raise ValueError("No target drug response available after filtering.")
    if bool(cfg["target"].get("require_all_drugs", True)):
        drug_wide = drug_wide.dropna(subset=keep_drugs)

    endpoint = drug_wide[keep_drugs].mean(axis=1).rename(target_col)
    feature_tbl = work.groupby(id_col, as_index=True)[feature_cols].median()
    merged = feature_tbl.join(endpoint, how="inner").reset_index()
    merged = merged.dropna(subset=[target_col]).copy()
    merged = merged.drop_duplicates(subset=[id_col]).copy()

    if len(merged) < 8:
        raise ValueError("Not enough rows after endpoint construction. Need at least 8 rows.")
    return merged, feature_cols


def run_stage_preprocess(
    config_path: str | Path,
    generate_demo_data: bool = False,
    force_demo_data: bool = False,
) -> StageResult:
    cfg = load_config(config_path)
    paths = _artifact_paths(cfg)
    _ensure_dirs(paths)

    input_path = _resolve(cfg, cfg["paths"]["input_csv"])
    if generate_demo_data:
        input_path = generate_demo_input(cfg, force=force_demo_data)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    raw = pd.read_csv(input_path)
    endpoint_df, feature_cols = _prepare_endpoint_table(raw, cfg)

    id_col = cfg["data"]["id_col"]
    target_col = cfg["target"]["name"]
    train_ratio = float(cfg["split"]["train_ratio"])
    random_state = int(cfg["split"]["random_state"])

    train_df, test_df = train_test_split(
        endpoint_df,
        train_size=train_ratio,
        random_state=random_state,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    train_scaled.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    test_scaled.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    train_scaled.to_csv(paths["train_processed"], index=False, encoding="utf-8")
    test_scaled.to_csv(paths["test_processed"], index=False, encoding="utf-8")
    train_df.to_csv(paths["train_unscaled"], index=False, encoding="utf-8")
    test_df.to_csv(paths["test_unscaled"], index=False, encoding="utf-8")
    save_json(paths["feature_columns"], {"id_col": id_col, "target_col": target_col, "features": feature_cols})
    save_json(
        paths["preprocess_meta"],
        {
            "random_state": random_state,
            "train_ratio": train_ratio,
            "input_rows": int(len(raw)),
            "endpoint_rows": int(len(endpoint_df)),
            "train_rows": int(len(train_scaled)),
            "test_rows": int(len(test_scaled)),
            "target_drugs": cfg["target"]["drugs"],
            "target_definition": "mean(log2(IC50)_trastuzumab, log2(IC50)_lapatinib)",
            "preprocessing_scope": "scaler fit on train split only; test split transformed with train scaler",
        },
    )
    return StageResult(
        message="preprocess completed",
        files=[str(paths["train_processed"]), str(paths["test_processed"]), str(paths["preprocess_meta"])],
    )


def run_stage_bootstrap(config_path: str | Path) -> StageResult:
    cfg = load_config(config_path)
    paths = _artifact_paths(cfg)
    _ensure_dirs(paths)

    train = pd.read_csv(paths["train_processed"])
    target_n = int(cfg["bootstrap"]["target_n"])
    seed = int(cfg["project"]["random_seed"])

    if len(train) == 0:
        raise ValueError("train_processed.csv is empty.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(train.index.to_numpy(), size=target_n, replace=True)
    boot = train.loc[idx].reset_index(drop=True).copy()
    boot["bootstrap_source_index"] = idx
    boot.to_csv(paths["bootstrap_train"], index=False, encoding="utf-8")

    return StageResult(message="bootstrap completed", files=[str(paths["bootstrap_train"])])


def _rf_params_from_config(cfg: Dict, strategy: str) -> List[Dict[str, int]]:
    if strategy == "paper_best":
        return [dict(cfg["model_selection"]["paper_best"])]

    grid = cfg["model_selection"]["rf_grid"]
    combos = itertools.product(grid["n_estimators"], grid["max_depth"], grid["min_samples_split"])
    return [
        {
            "n_estimators": int(n),
            "max_depth": int(d),
            "min_samples_split": int(m),
        }
        for n, d, m in combos
    ]


def _pick_best_row(df: pd.DataFrame) -> pd.Series:
    rank = df.sort_values(["composite_score", "cv_r2_mean", "oob_r2"], ascending=False).reset_index(drop=True)
    return rank.iloc[0]


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "R2": r2}


def run_stage_train_rf(config_path: str | Path, strategy: str | None = None) -> StageResult:
    cfg = load_config(config_path)
    paths = _artifact_paths(cfg)
    _ensure_dirs(paths)

    feat_info = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    feature_cols = feat_info["features"]
    target_col = feat_info["target_col"]

    train = pd.read_csv(paths["bootstrap_train"])
    test = pd.read_csv(paths["test_processed"])
    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    seed = int(cfg["project"]["random_seed"])
    cv_folds = int(cfg["model_selection"]["cv_folds"])
    cv_folds = max(2, min(cv_folds, len(train)))
    rf_n_jobs = int(cfg["model_selection"].get("rf_n_jobs", 1))

    use_strategy = strategy or cfg["model_selection"]["strategy"]
    params_grid = _rf_params_from_config(cfg, use_strategy)

    rows = []
    for params in params_grid:
        model = RandomForestRegressor(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            random_state=seed,
            oob_score=True,
            bootstrap=True,
            n_jobs=rf_n_jobs,
        )
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        cv_scores = cross_val_score(model, X_train, y_train, scoring="r2", cv=cv, n_jobs=1)
        model.fit(X_train, y_train)
        oob_r2 = float(model.oob_score_)
        cv_mean = float(np.mean(cv_scores))
        composite = float(np.mean([cv_mean, oob_r2]))
        rows.append(
            {
                "n_estimators": int(params["n_estimators"]),
                "max_depth": int(params["max_depth"]),
                "min_samples_split": int(params["min_samples_split"]),
                "cv_r2_mean": cv_mean,
                "oob_r2": oob_r2,
                "composite_score": composite,
            }
        )

    selection = pd.DataFrame(rows)
    best = _pick_best_row(selection)

    best_model = RandomForestRegressor(
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        min_samples_split=int(best["min_samples_split"]),
        random_state=seed,
        oob_score=True,
        bootstrap=True,
        n_jobs=rf_n_jobs,
    )
    best_model.fit(X_train, y_train)
    pred = best_model.predict(X_test)
    metric = _metrics(y_test, pred)

    selection.to_csv(paths["rf_selection"], index=False, encoding="utf-8")
    save_json(
        paths["rf_metrics"],
        {
            "strategy": use_strategy,
            "cv_folds": cv_folds,
            "best_params": {
                "n_estimators": int(best["n_estimators"]),
                "max_depth": int(best["max_depth"]),
                "min_samples_split": int(best["min_samples_split"]),
            },
            "best_cv_r2": float(best["cv_r2_mean"]),
            "best_oob_r2": float(best["oob_r2"]),
            "test_metrics": metric,
            "paper_optimal_reference": cfg["model_selection"]["paper_best"],
        },
    )
    joblib.dump(best_model, paths["rf_model"])

    return StageResult(
        message="rf training completed",
        files=[str(paths["rf_model"]), str(paths["rf_selection"]), str(paths["rf_metrics"])],
    )


def run_stage_baselines(config_path: str | Path) -> StageResult:
    cfg = load_config(config_path)
    paths = _artifact_paths(cfg)
    _ensure_dirs(paths)

    feat_info = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    id_col = feat_info["id_col"]
    target_col = feat_info["target_col"]
    feature_cols = feat_info["features"]

    train = pd.read_csv(paths["bootstrap_train"])
    test = pd.read_csv(paths["test_processed"])

    X_train = train[feature_cols].to_numpy(dtype=float)
    y_train = train[target_col].to_numpy(dtype=float)
    X_test = test[feature_cols].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    rf = joblib.load(paths["rf_model"])
    pred_rf = rf.predict(X_test)
    metric_rf = _metrics(y_test, pred_rf)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    metric_lr = _metrics(y_test, pred_lr)

    svr_cfg = cfg["baselines"]["svr_rbf"]
    svr = SVR(kernel="rbf", C=float(svr_cfg["C"]), gamma=svr_cfg["gamma"])
    svr.fit(X_train, y_train)
    pred_svr = svr.predict(X_test)
    metric_svr = _metrics(y_test, pred_svr)

    pred_df = pd.DataFrame(
        {
            id_col: test[id_col].astype(str).tolist(),
            "y_true": y_test,
            "rf_pred": pred_rf,
            "lr_pred": pred_lr,
            "svr_pred": pred_svr,
        }
    )
    pred_df.to_csv(paths["predictions"], index=False, encoding="utf-8")

    save_json(
        paths["baseline_metrics"],
        {
            "target": target_col,
            "random_state": int(cfg["project"]["random_seed"]),
            "models": {
                "RandomForest": metric_rf,
                "LinearRegression": metric_lr,
                "SVR_RBF": metric_svr,
            },
        },
    )
    return StageResult(message="baselines completed", files=[str(paths["baseline_metrics"]), str(paths["predictions"])])


def _connected_components(nodes: Sequence[str], corr: pd.DataFrame, threshold: float) -> List[List[str]]:
    graph = {n: set() for n in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1 :]:
            val = float(corr.loc[a, b])
            if np.isnan(val):
                continue
            if abs(val) > threshold:
                graph[a].add(b)
                graph[b].add(a)

    seen = set()
    comps: List[List[str]] = []
    for n in nodes:
        if n in seen:
            continue
        stack = [n]
        seen.add(n)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comps.append(sorted(comp))
    return comps


def run_stage_feature_importance(config_path: str | Path, mode: str = "demo") -> StageResult:
    cfg = load_config(config_path)
    paths = _artifact_paths(cfg)
    _ensure_dirs(paths)

    feat_info = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    feature_cols = feat_info["features"]
    target_col = feat_info["target_col"]

    train = pd.read_csv(paths["bootstrap_train"])
    test = pd.read_csv(paths["test_processed"])
    X_train = train[feature_cols]
    y_train = train[target_col].to_numpy(dtype=float)
    X_test = test[feature_cols]
    y_test = test[target_col].to_numpy(dtype=float)

    rf = joblib.load(paths["rf_model"])
    impurity = pd.Series(rf.feature_importances_, index=feature_cols, dtype=float)

    repeats = int(cfg["feature_importance"]["permutation_repeats_demo"])
    if mode != "demo":
        repeats = int(cfg["feature_importance"]["permutation_repeats_full"])
    perm = permutation_importance(
        rf,
        X_test.to_numpy(dtype=float),
        y_test,
        n_repeats=repeats,
        random_state=int(cfg["project"]["random_seed"]),
        scoring="r2",
        n_jobs=1,
    )
    perm_mean = pd.Series(perm.importances_mean, index=feature_cols, dtype=float)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "impurity_importance": impurity.reindex(feature_cols).values,
            "permutation_importance": perm_mean.reindex(feature_cols).values,
        }
    )
    importance["impurity_norm"] = minmax_norm(importance["impurity_importance"])
    importance["permutation_norm"] = minmax_norm(importance["permutation_importance"])
    importance["combined_importance"] = 0.5 * importance["impurity_norm"] + 0.5 * importance["permutation_norm"]
    importance = importance.sort_values("combined_importance", ascending=False).reset_index(drop=True)
    importance.to_csv(paths["feature_importance"], index=False, encoding="utf-8")

    threshold = float(cfg["feature_importance"]["spearman_threshold"])
    corr = X_train.corr(method="spearman").fillna(0.0)
    comps = _connected_components(feature_cols, corr, threshold)

    score_map = importance.set_index("feature")["combined_importance"].to_dict()
    rep_rows = []
    cluster_rows = []
    for cid, comp in enumerate(comps, start=1):
        rep = max(comp, key=lambda x: score_map.get(x, float("-inf")))
        rep_rows.append(
            {
                "cluster_id": cid,
                "cluster_size": len(comp),
                "representative_feature": rep,
                "representative_combined_importance": float(score_map.get(rep, np.nan)),
                "cluster_members": ";".join(comp),
            }
        )
        for feat in comp:
            cluster_rows.append(
                {
                    "cluster_id": cid,
                    "feature": feat,
                    "is_representative": int(feat == rep),
                }
            )

    rep_df = pd.DataFrame(rep_rows).sort_values("representative_combined_importance", ascending=False)
    cluster_df = pd.DataFrame(cluster_rows).sort_values(["cluster_id", "feature"])
    rep_df.to_csv(paths["representatives"], index=False, encoding="utf-8")
    cluster_df.to_csv(paths["clusters"], index=False, encoding="utf-8")

    pred = pd.read_csv(paths["predictions"])
    _plot_outputs(pred, importance, paths)
    _write_run_summary(cfg, paths)

    return StageResult(
        message="feature importance completed",
        files=[str(paths["feature_importance"]), str(paths["representatives"]), str(paths["summary"])],
    )


def _plot_outputs(pred: pd.DataFrame, importance: pd.DataFrame, paths: Dict[str, Path]) -> None:
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["rf_pred"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(y_true, y_pred, alpha=0.75, edgecolors="none")
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.1)
    ax.set_xlabel("Observed combined log2(IC50)")
    ax.set_ylabel("Predicted combined log2(IC50)")
    ax.set_title("Demo RF Prediction")
    fig.tight_layout()
    fig.savefig(paths["fig_scatter"], dpi=180)
    plt.close(fig)

    top = importance.head(10).copy()
    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    ax2.barh(top["feature"][::-1], top["combined_importance"][::-1], color="#2a9d8f")
    ax2.set_xlabel("Combined importance")
    ax2.set_title("Top-10 Feature Importance (Demo)")
    fig2.tight_layout()
    fig2.savefig(paths["fig_importance"], dpi=180)
    plt.close(fig2)


def _write_run_summary(cfg: Dict, paths: Dict[str, Path]) -> None:
    rf_metrics = json.loads(paths["rf_metrics"].read_text(encoding="utf-8"))
    base_metrics = json.loads(paths["baseline_metrics"].read_text(encoding="utf-8"))
    reps = pd.read_csv(paths["representatives"]).head(10)
    lines = [
        "# Lightweight Demo Summary",
        "",
        "This run is for reproducibility workflow validation only.",
        "",
        "## RF Selection",
        f"- Strategy: {rf_metrics['strategy']}",
        (
            "- Best params: "
            f"n_estimators={rf_metrics['best_params']['n_estimators']}, "
            f"max_depth={rf_metrics['best_params']['max_depth']}, "
            f"min_samples_split={rf_metrics['best_params']['min_samples_split']}"
        ),
        f"- 5-fold CV R2: {rf_metrics['best_cv_r2']:.4f}",
        f"- OOB R2: {rf_metrics['best_oob_r2']:.4f}",
        f"- Test R2: {rf_metrics['test_metrics']['R2']:.4f}",
        f"- Test RMSE: {rf_metrics['test_metrics']['RMSE']:.4f}",
        "",
        "## Baselines",
        f"- LinearRegression R2: {base_metrics['models']['LinearRegression']['R2']:.4f}",
        f"- SVR(RBF) R2: {base_metrics['models']['SVR_RBF']['R2']:.4f}",
        "",
        "## Top Representative Features",
    ]
    for _, row in reps.iterrows():
        lines.append(
            f"- Cluster {int(row['cluster_id'])}: {row['representative_feature']} "
            f"(combined={float(row['representative_combined_importance']):.4f})"
        )
    lines += [
        "",
        "## Notes",
        "- Default run does not execute heavy full-grid search.",
        "- Enrichment analysis is kept as interface/documentation only.",
    ]
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo_pipeline(config_path: str | Path, force_demo_data: bool = False) -> List[StageResult]:
    results = []
    results.append(run_stage_preprocess(config_path, generate_demo_data=True, force_demo_data=force_demo_data))
    results.append(run_stage_bootstrap(config_path))
    results.append(run_stage_train_rf(config_path))
    results.append(run_stage_baselines(config_path))
    results.append(run_stage_feature_importance(config_path, mode="demo"))
    return results
