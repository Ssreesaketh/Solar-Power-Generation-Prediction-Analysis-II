"""
Model Explainability and Diagnostic Analysis Module.
Extracts feature importances, permutation importances, residual diagnostic plots,
and analyzes meteorological associations with operational deviations.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sns.set_theme(style="whitegrid", palette="muted")
VIS_DIR = Path("ml/outputs/visualizations")
VIS_DIR.mkdir(parents=True, exist_ok=True)


def extract_feature_importance(
    model: object,
    feature_names: List[str],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Extracts native feature importance from tree/ensemble model,
    or falls back to permutation importance if model doesn't expose MDI (e.g., HistGradientBoosting or Voting).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps["model"], "feature_importances_"):
        importances = model.named_steps["model"].feature_importances_
    elif X_val is not None and y_val is not None:
        perm = compute_permutation_importance(model, X_val, y_val, n_repeats=3)
        return perm.rename(columns={"Permutation_Importance_Mean": "Importance"})[["Feature", "Importance"]]
    else:
        # Equal weighting fallback
        importances = np.ones(len(feature_names)) / len(feature_names)

    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    return df_imp


def compute_permutation_importance(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes model-agnostic permutation feature importance on the validation set.
    """
    r = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    df_perm = pd.DataFrame({
        "Feature": X_val.columns,
        "Permutation_Importance_Mean": r.importances_mean,
        "Permutation_Importance_Std": r.importances_std,
    }).sort_values(by="Permutation_Importance_Mean", ascending=False).reset_index(drop=True)
    return df_perm


def generate_evaluation_plots(
    test_df: pd.DataFrame,
    df_importance: pd.DataFrame,
    comparison_df: pd.DataFrame,
    y_true_col: str = "AC_POWER",
    y_pred_col: str = "pred_AC_POWER",
):
    """
    Generates 8 diagnostic evaluation plots:
      1. Actual vs Predicted scatter
      2. Residual distribution & Q-Q plot
      3. Residual vs Predicted
      4. Prediction error over time
      5. Feature importance
      6. Plant-wise comparison
      7. Daylight vs nighttime evaluation
      8. Model comparison chart
    """
    y_true = test_df[y_true_col]
    y_pred = test_df[y_pred_col]
    residuals = y_true - y_pred

    # 1. Actual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_true, y_pred, alpha=0.4, color="#2b5c8f", s=18, edgecolors="none")
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Ideal 1:1 Reference")
    ax.set_title("Actual vs. Predicted AC Power (Untouched Test Set: June 13-17, 2020)")
    ax.set_xlabel("Actual AC Power (kW)")
    ax.set_ylabel("Predicted AC Power (kW)")
    ax.set_xlim(-50, max_val)
    ax.set_ylim(-50, max_val)
    ax.legend()
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_01_actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 2. Residual Distribution Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals, bins=40, kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].axvline(0, color="r", linestyle="--", lw=1.5)
    axes[0].set_title(f"Residual Distribution (Mean={residuals.mean():.1f} kW, Std={residuals.std():.1f} kW)")
    axes[0].set_xlabel("Residual (Actual - Predicted AC Power kW)")
    axes[0].set_ylabel("Count")

    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q Plot of Residuals")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_02_residual_distribution.png", dpi=200)
    plt.close(fig)

    # 3. Residual vs Predicted Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y_pred, residuals, alpha=0.4, color="#e07a5f", s=18)
    ax.axhline(0, color="k", linestyle="--", lw=1.5)
    ax.set_title("Residual vs. Predicted AC Power")
    ax.set_xlabel("Predicted AC Power (kW)")
    ax.set_ylabel("Residual (Actual - Predicted kW)")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_03_residual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 4. Prediction Error Over Time
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pd.to_datetime(test_df["DATE_TIME"]), residuals, color="#3d405b", lw=1.2, label="Residual (kW)")
    ax.axhline(0, color="k", linestyle="--", lw=1.2)
    ax.set_title("Prediction Error Over Time (Untouched Test Period)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual (kW)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_04_error_over_time.png", dpi=200)
    plt.close(fig)

    # 5. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_importance, x="Importance", y="Feature", ax=ax, palette="mako")
    ax.set_title("Feature Importance Ranking")
    ax.set_xlabel("Relative Predictive Importance")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_05_feature_importance.png", dpi=200)
    plt.close(fig)

    # 6. Plant-wise Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, plant_id in enumerate([4135001, 4136001]):
        p_name = "Plant 1" if plant_id == 4135001 else "Plant 2"
        p_mask = test_df["PLANT_ID"] == plant_id
        axes[i].scatter(test_df.loc[p_mask, y_true_col], test_df.loc[p_mask, y_pred_col], alpha=0.5, s=20, color="#2b5c8f" if i == 0 else "#e07a5f")
        p_max = max(test_df.loc[p_mask, y_true_col].max(), test_df.loc[p_mask, y_pred_col].max()) * 1.05
        axes[i].plot([0, p_max], [0, p_max], "r--")
        axes[i].set_title(f"{p_name}: Actual vs Predicted")
        axes[i].set_xlabel("Actual (kW)")
        axes[i].set_ylabel("Predicted (kW)")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_06_plant_comparison.png", dpi=200)
    plt.close(fig)

    # 7. Daylight vs Nighttime Evaluation
    day_mask = test_df["IRRADIATION"] > 0
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(residuals[day_mask], ax=axes[0], color="#f4a261", kde=True, bins=30)
    axes[0].set_title(f"Daylight Residuals (N={day_mask.sum()}, Mean={residuals[day_mask].mean():.1f} kW)")
    axes[0].set_xlabel("Residual (kW)")

    sns.histplot(residuals[~day_mask], ax=axes[1], color="#264653", kde=True, bins=20)
    axes[1].set_title(f"Nighttime Residuals (N={(~day_mask).sum()}, Mean={residuals[~day_mask].mean():.1f} kW)")
    axes[1].set_xlabel("Residual (kW)")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_07_daylight_vs_nighttime.png", dpi=200)
    plt.close(fig)

    # 8. Model Comparison Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    comp_sorted = comparison_df.sort_values(by="Val_RMSE", ascending=True)
    sns.barplot(data=comp_sorted, x="Val_RMSE", y="Model", ax=ax, palette="crest_r")
    ax.set_title(f"Algorithm Benchmark: Validation RMSE (kW) Across {len(comp_sorted)} Candidate Regressors")
    ax.set_xlabel("Validation RMSE (kW) [Lower is Better]")
    plt.tight_layout()
    fig.savefig(VIS_DIR / "eval_08_model_comparison_chart.png", dpi=200)
    plt.close(fig)

    print("All diagnostic evaluation plots generated successfully.")
