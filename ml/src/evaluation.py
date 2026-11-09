"""
Model Evaluation Module.
Computes standard regression metrics (MAE, MSE, RMSE, R², NRMSE),
subgroup performance (daylight vs nighttime, plant-wise, power-tier),
and generates diagnostic residual plots.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Computes MAE, MSE, RMSE, R2, and NRMSE (range-normalized).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Solar power is physically non-negative
    y_pred_clipped = np.clip(y_pred, a_min=0.0, a_max=None)

    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    mse = float(mean_squared_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred_clipped))

    y_range = float(np.ptp(y_true))
    nrmse_range = float(rmse / y_range) if y_range > 0 else 0.0

    mean_val = float(np.mean(y_true))
    nrmse_mean = float(rmse / mean_val) if mean_val > 0 else 0.0

    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}mae": round(mae, 4),
        f"{p}mse": round(mse, 4),
        f"{p}rmse": round(rmse, 4),
        f"{p}r2": round(r2, 4),
        f"{p}nrmse_range": round(nrmse_range, 4),
        f"{p}nrmse_mean": round(nrmse_mean, 4),
        f"{p}samples": len(y_true),
    }


def compute_subgroup_metrics(
    df: pd.DataFrame,
    y_true_col: str = "AC_POWER",
    y_pred_col: str = "pred_AC_POWER",
) -> Dict[str, Dict[str, float]]:
    """
    Computes metrics across operational partitions:
      - overall
      - daylight (IRRADIATION > 0)
      - nighttime (IRRADIATION == 0)
      - Plant 1 (PLANT_ID == 4135001)
      - Plant 2 (PLANT_ID == 4136001)
      - Power tiers: Low (<20% max), Medium (20-70%), High (>70%)
    """
    results = {}
    y_t = df[y_true_col].values
    y_p = df[y_pred_col].values

    # Overall
    results["overall"] = compute_metrics(y_t, y_p)

    # Daylight vs Night
    if "IRRADIATION" in df.columns:
        day_mask = df["IRRADIATION"] > 0
        night_mask = ~day_mask
        if day_mask.sum() > 0:
            results["daylight"] = compute_metrics(y_t[day_mask], y_p[day_mask])
        if night_mask.sum() > 0:
            results["nighttime"] = compute_metrics(y_t[night_mask], y_p[night_mask])

    # Plant-wise
    if "PLANT_ID" in df.columns:
        for plant_id in df["PLANT_ID"].unique():
            p_mask = df["PLANT_ID"] == plant_id
            p_name = "plant_1" if plant_id == 4135001 else "plant_2"
            results[p_name] = compute_metrics(y_t[p_mask], y_p[p_mask])

    # Power tiers (daylight only)
    max_p = np.max(y_t)
    if max_p > 0:
        low_mask = y_t <= (0.20 * max_p)
        med_mask = (y_t > 0.20 * max_p) & (y_t <= 0.70 * max_p)
        high_mask = y_t > (0.70 * max_p)

        if low_mask.sum() > 0:
            results["tier_low_power"] = compute_metrics(y_t[low_mask], y_p[low_mask])
        if med_mask.sum() > 0:
            results["tier_med_power"] = compute_metrics(y_t[med_mask], y_p[med_mask])
        if high_mask.sum() > 0:
            results["tier_high_power"] = compute_metrics(y_t[high_mask], y_p[high_mask])

    return results
