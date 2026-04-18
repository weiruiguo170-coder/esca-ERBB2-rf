from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "r2": r2}


def evaluate_prediction_table(pred_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y_true = pred_table["y_true"].to_numpy()

    for model_name, pred_col in [
        ("rf", "rf_pred"),
        ("lr", "lr_pred"),
        ("svr", "svr_pred"),
    ]:
        metrics = regression_metrics(y_true, pred_table[pred_col].to_numpy())
        rows.append({"model": model_name, "rmse": metrics["rmse"], "r2": metrics["r2"]})

    return pd.DataFrame(rows).sort_values("r2", ascending=False)
