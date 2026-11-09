# Solar Power Generation Prediction & Performance Deviation Analysis: Phase I (ML)

## Overview
This directory contains the complete Machine Learning system for predicting solar plant power generation and diagnosing operational deviations using 34 days of 15-minute interval solar telemetry (May 15 to June 17, 2020) across Plant 1 (ID: 4135001) and Plant 2 (ID: 4136001).

## Key Architecture & Components

```
ml/
├── data/
│   ├── raw/                 # Plant 1 & 2 Generation and Weather CSVs
│   └── processed/           # Cached processed partitions
├── notebooks/               # Jupyter research and tutorial walkthroughs
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_deviation_analysis.ipynb
├── src/
│   ├── data_loader.py       # Timestamp format-aware data loading
│   ├── data_validation.py   # Physical sensor boundary checks
│   ├── preprocessing.py     # Reusable Scikit-learn cleaner
│   ├── feature_engineering.py # Cyclical diurnal, thermal & solar interaction features
│   ├── model_training.py    # Chronological training, TimeSeriesSplit CV
│   ├── model_selection.py   # 12-regressor benchmark suite
│   ├── evaluation.py        # Subgroup metrics (day/night, plant-wise, power tiers)
│   ├── prediction.py        # Production predictor wrapper with physical solar gating
│   ├── deviation_detection.py # Dynamic MAD & irradiance-scaled anomaly classifier
│   └── explainability.py    # MDI, permutation importance & diagnostic residual plots
├── models/
│   ├── best_model.joblib    # Voting Regressor ensemble (HistGradientBoosting + LightGBM + GradientBoosting)
│   ├── preprocessing_pipeline.joblib # Fitted feature engineer
│   ├── deviation_detector.joblib     # Fitted anomaly detector
│   ├── deviation_thresholds.json     # MAD & severity cutoffs
│   └── model_metadata.json  # Comprehensive model manifest & evaluation history
├── reports/
│   ├── eda_report.md        # Comprehensive exploratory analysis
│   ├── model_comparison.csv # Actual measured metrics across 12 candidate algorithms
│   ├── evaluation_metrics.json # Test metrics on untouched June 13-17 split
│   ├── deviation_summary.csv   # Flagged operational deviations
│   └── experiment_summary.md   # Chronological validation & scientific justification
├── outputs/
│   └── visualizations/      # 16 publication-grade diagnostic charts
├── tests/
│   └── test_ml_pipeline.py  # Pytest test suite
└── requirements.txt
```

## Performance Highlights (Untouched Test Period: June 13 - 17, 2020)
- **Overall $R^2$**: 0.9602
- **Overall MAE**: 587.21 kW
- **Plant 1 $R^2$**: 0.9903 (extremely tight physical tracking)
- **Plant 2 $R^2$**: 0.8928 (captures localized inverter trips and thermal derating)
- **Physical Solar Gating**: Enforces 0.0 kW nocturnal power generation, eliminating nocturnal model drift.

## Running Tests
```bash
python -m pytest ml/tests/ -v
```
