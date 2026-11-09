# Experiment & Model Evaluation Summary
## Solar Power Generation Prediction and Performance Deviation Analysis

### 1. Chronological Partitioning Strategy
To prevent temporal data leakage and respect physical causality, the 34 days of operational telemetry were partitioned chronologically:
- **Training Set (70%)**: 2020-05-15 00:00:00 to 2020-06-07 19:00:00 (4,460 observations)
- **Validation Set (15%)**: 2020-06-07 19:15:00 to 2020-06-12 21:15:00 (978 observations)
- **Untouched Test Set (15%)**: 2020-06-12 21:30:00 to 2020-06-17 23:45:00 (978 observations)

All preprocessing transformations, feature scalers, and deviation detection baselines were strictly fitted on the training split only.

### 2. Algorithm Suitability Analysis & Benchmarking
Candidate regression algorithms spanning baseline, linear, regularized, tree, ensemble, kernel, instance-based, gradient-boosted, and stacked architectures were benchmarked on the validation set:

```
| Model                    |   Val_MAE |   Val_RMSE |   Val_R2 |   Val_NRMSE_Range |   Train_R2 |   Train_Time_s |   Infer_Time_ms | Overfitting_Risk       |
|:-------------------------|----------:|-----------:|---------:|------------------:|-----------:|---------------:|----------------:|:-----------------------|
| Voting Ensemble          |   442.339 |    1230.11 |   0.9702 |            0.0433 |     0.9955 |          3.918 |           41.67 | Low (Well-Regularized) |
| HistGradientBoosting     |   466.368 |    1240.87 |   0.9696 |            0.0437 |     0.9943 |          0.631 |           15.95 | Low (Well-Regularized) |
| LightGBM                 |   436.801 |    1244.29 |   0.9695 |            0.0438 |     0.9961 |          0.236 |           17.15 | Low (Well-Regularized) |
| Gradient Boosting        |   470.569 |    1279.85 |   0.9677 |            0.045  |     0.9953 |          3.757 |            5.96 | Low (Well-Regularized) |
| Random Forest            |   477.885 |    1312    |   0.9661 |            0.0462 |     0.9963 |          1.942 |           52.34 | Low (Well-Regularized) |
| Stacking Regressor       |   514.652 |    1328.03 |   0.9652 |            0.0467 |     0.9939 |          5.502 |           66.84 | Low (Well-Regularized) |
| Extra Trees              |   484.256 |    1359.74 |   0.9635 |            0.0478 |     0.9982 |          0.872 |           62.04 | Low (Well-Regularized) |
| K-Nearest Neighbors      |   624.798 |    1412.96 |   0.9606 |            0.0497 |     0.9854 |          0.033 |           34.2  | Low (Well-Regularized) |
| Decision Tree            |   583.082 |    1604.77 |   0.9492 |            0.0565 |     0.9937 |          0.059 |            2.91 | Low (Well-Regularized) |
| Linear Regression        |   783.5   |    1743.84 |   0.94   |            0.0613 |     0.9322 |          0.012 |            3.19 | Low (Well-Regularized) |
| Lasso Regression         |   859.54  |    1803.56 |   0.9359 |            0.0635 |     0.9295 |          0.172 |            3.33 | Low (Well-Regularized) |
| Ridge Regression         |   904.099 |    1850.73 |   0.9325 |            0.0651 |     0.9276 |          0.009 |            2.74 | Low (Well-Regularized) |
| Support Vector Regressor |   922.78  |    1900.91 |   0.9287 |            0.0669 |     0.9333 |          1.396 |          553.86 | Low (Well-Regularized) |
| Dummy (Baseline)         |  3625.38  |    5905.39 |   0.3123 |            0.2078 |     0.2625 |          0.056 |            0.84 | Low (Well-Regularized) |
```

### 3. Hyperparameter Tuning & Architecture Selection
The top-performing algorithm (**Solar_Power_Predictor_Voting_Ensemble**) was tuned using `TimeSeriesSplit` (5 chronological folds) over the training partition.
- **Model Version**: v1.1.0
- **Features Used (15)**: IRRADIATION, AMBIENT_TEMPERATURE, MODULE_TEMPERATURE, temp_diff, irr_mod_interaction, irr_amb_interaction, irradiation_squared, sin_hour, cos_hour, hour, day_of_week, day_of_year, is_plant_2, effective_irradiance, solar_elevation_proxy

### 4. Untouched Test Set Evaluation
Evaluated strictly on the unseen test period (2020-06-12 21:30:00 to 2020-06-17 23:45:00):

| Subgroup / Slice | MAE (kW) | RMSE (kW) | R² | NRMSE (Range) | Sample Count |
|---|---|---|---|---|---|
| **Overall** | 486.02 | 1341.10 | 0.9640 | 0.046 | 978 |
| **Daylight Only** | 886.80 | 1811.55 | 0.9324 | 0.062 | 536 |
| **Nighttime (Gated)** | 0.00 | 0.00 | 1.0000 | 0.000 | 442 |
| **Plant 1 (ID 4135001)** | 277.34 | 647.85 | 0.9937 | 0.022 | 488 |
| **Plant 2 (ID 4136001)** | 693.84 | 1780.95 | 0.9000 | 0.079 | 490 |

### 5. Deviation Analysis Findings
Using Median Absolute Deviation (MAD = 259.7 kW) with dynamic irradiance scaling:
- **Normal Operations**: 659 records (67.4%)
- **Moderate Deviations**: 93 records (9.5%)
- **Significant Deviations**: 80 records (8.2%)
- **Extreme Deviations**: 146 records (14.9%)

Significant and Extreme deviations concentrated predominantly in **Plant 2**, directly reflecting localized inverter curtailment (e.g. 18/22 active inverters) and elevated midday panel temperatures (> 55°C).
