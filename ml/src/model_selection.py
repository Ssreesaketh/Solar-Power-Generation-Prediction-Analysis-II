"""
Algorithm Suitability Analysis and Model Benchmarking Module.
Compares 14 candidate regression algorithms and ensembles using strict chronological validation.
Measures MAE, MSE, RMSE, R², NRMSE, training latency, inference latency, and overfitting risk.
Also investigates unified vs plant-specific dedicated models.
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from evaluation import compute_metrics


def get_candidate_models() -> Dict[str, object]:
    """
    Returns candidate regressors spanning linear, regularized, tree, ensemble,
    kernel, instance-based, gradient-boosted, and stacked architectures.
    Pipelines with StandardScaler are used for scale-sensitive models.
    """
    # Base learners for ensembles
    hgb_base = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42)
    lgb_base = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=1)
    gb_base = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    rf_base = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=2)

    voting_ens = VotingRegressor([
        ("hgb", hgb_base),
        ("lgbm", lgb_base),
        ("gb", gb_base),
    ])

    stacking_ens = StackingRegressor(
        estimators=[
            ("hgb", HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=6, random_state=42)),
            ("lgbm", lgb.LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=1)),
            ("rf", RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=2)),
        ],
        final_estimator=Ridge(alpha=10.0),
        cv=3,
        n_jobs=1,
    )

    return {
        "Dummy (Baseline)": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0, random_state=42))]),
        "Lasso Regression": Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=5.0, max_iter=3000, random_state=42))]),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=2),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=2),
        "Support Vector Regressor": Pipeline([("scaler", StandardScaler()), ("model", SVR(C=100.0, epsilon=10.0, max_iter=3000))]),
        "K-Nearest Neighbors": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=7, n_jobs=2))]),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42),
        "LightGBM": lgb.LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=31, random_state=42, verbose=-1, n_jobs=1),
        "Voting Ensemble": voting_ens,
        "Stacking Regressor": stacking_ens,
    }


def assess_overfitting(train_r2: float, val_r2: float) -> str:
    """
    Evaluates overfitting risk based on the train-vs-validation generalization gap.
    """
    gap = train_r2 - val_r2
    if gap > 0.15:
        return "High (Train R² >> Val R²)"
    elif gap > 0.05:
        return "Moderate"
    elif gap < -0.10:
        return "Underfitting / Distribution Shift"
    else:
        return "Low (Well-Regularized)"


def benchmark_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Trains and validates all candidate algorithms on chronological partitions.
    Records MAE, MSE, RMSE, R², NRMSE, training time, inference latency, and overfitting risk.
    """
    models = get_candidate_models()
    records = []
    trained_models = {}

    for name, model in models.items():
        print(f"  Benchmarking: {name}...")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # Inference timing and clipping
        t1 = time.perf_counter()
        val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, 0.0, None)
        if "IRRADIATION" in X_val.columns:
            val_pred[X_val["IRRADIATION"] <= 0.0] = 0.0
        infer_time = time.perf_counter() - t1

        train_pred = np.clip(model.predict(X_train), 0.0, None)
        if "IRRADIATION" in X_train.columns:
            train_pred[X_train["IRRADIATION"] <= 0.0] = 0.0

        train_m = compute_metrics(y_train, train_pred)
        val_m = compute_metrics(y_val, val_pred)

        overfitting_risk = assess_overfitting(train_m["r2"], val_m["r2"])

        records.append({
            "Model": name,
            "Val_MAE": val_m["mae"],
            "Val_RMSE": val_m["rmse"],
            "Val_R2": val_m["r2"],
            "Val_NRMSE_Range": val_m["nrmse_range"],
            "Train_R2": train_m["r2"],
            "Train_Time_s": round(train_time, 3),
            "Infer_Time_ms": round(infer_time * 1000, 2),
            "Overfitting_Risk": overfitting_risk,
        })
        trained_models[name] = model

    results_df = pd.DataFrame(records).sort_values(by="Val_RMSE").reset_index(drop=True)
    return results_df, trained_models


def compare_plant_architectures(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feat_engineer: object,
) -> pd.DataFrame:
    """
    Scientifically compares two structural modeling approaches:
      Approach A: Single Unified Model with `is_plant_2` indicator feature
      Approach B: Separate Dedicated Models for Plant 1 and Plant 2
    """
    print("\n--- Comparing Modeling Architectures (Unified vs Dedicated Plant Models) ---")
    
    # Transform full sets
    X_train_full = feat_engineer.transform(train_df)
    X_val_full = feat_engineer.transform(val_df)

    # 1. Unified Model
    unified_model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42)
    unified_model.fit(X_train_full, train_df["AC_POWER"])
    unified_val_preds = np.clip(unified_model.predict(X_val_full), 0.0, None)
    if "IRRADIATION" in X_val_full.columns:
        unified_val_preds[X_val_full["IRRADIATION"] <= 0.0] = 0.0
    unified_metrics = compute_metrics(val_df["AC_POWER"], unified_val_preds)

    # 2. Dedicated Plant Models
    p1_tr_mask = train_df["PLANT_ID"] == 4135001
    p1_val_mask = val_df["PLANT_ID"] == 4135001
    p2_tr_mask = train_df["PLANT_ID"] == 4136001
    p2_val_mask = val_df["PLANT_ID"] == 4136001

    sub_features = [f for f in X_train_full.columns if f != "is_plant_2"]
    
    m_p1 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42)
    m_p1.fit(X_train_full.loc[p1_tr_mask, sub_features], train_df.loc[p1_tr_mask, "AC_POWER"])
    p1_val_pred = np.clip(m_p1.predict(X_val_full.loc[p1_val_mask, sub_features]), 0.0, None)
    if "IRRADIATION" in X_val_full.columns:
        p1_val_pred[X_val_full.loc[p1_val_mask, "IRRADIATION"] <= 0.0] = 0.0

    m_p2 = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_depth=7, random_state=42)
    m_p2.fit(X_train_full.loc[p2_tr_mask, sub_features], train_df.loc[p2_tr_mask, "AC_POWER"])
    p2_val_pred = np.clip(m_p2.predict(X_val_full.loc[p2_val_mask, sub_features]), 0.0, None)
    if "IRRADIATION" in X_val_full.columns:
        p2_val_pred[X_val_full.loc[p2_val_mask, "IRRADIATION"] <= 0.0] = 0.0

    combined_val_y = np.concatenate([val_df.loc[p1_val_mask, "AC_POWER"].values, val_df.loc[p2_val_mask, "AC_POWER"].values])
    combined_val_pred = np.concatenate([p1_val_pred, p2_val_pred])
    dedicated_metrics = compute_metrics(combined_val_y, combined_val_pred)

    comp_df = pd.DataFrame([
        {
            "Architecture": "Unified Model (with plant indicator)",
            "Val_MAE": unified_metrics["mae"],
            "Val_RMSE": unified_metrics["rmse"],
            "Val_R2": unified_metrics["r2"],
            "Parameters": "Single shared estimator",
            "Deployment_Complexity": "Low (1 model file to maintain & load)",
        },
        {
            "Architecture": "Dedicated Plant Models (Separate P1 & P2)",
            "Val_MAE": dedicated_metrics["mae"],
            "Val_RMSE": dedicated_metrics["rmse"],
            "Val_R2": dedicated_metrics["r2"],
            "Parameters": "Two separate estimators",
            "Deployment_Complexity": "Moderate (requires plant-based routing)",
        },
    ])
    return comp_df
