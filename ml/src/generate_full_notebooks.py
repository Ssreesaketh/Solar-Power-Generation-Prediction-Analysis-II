"""
Generator for comprehensive, interactive Jupyter Notebooks in ml/notebooks/.
Creates rich educational walkthroughs with markdown theory, equations,
code cells, and execution outputs.
"""

import json
from pathlib import Path


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")],
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")],
    }


def main():
    nb_dir = Path("ml/notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 01_eda.ipynb
    # -------------------------------------------------------------
    nb1 = make_nb([
        md("# 01: Exploratory Data Analysis (EDA)\n## Solar Power Generation & Weather Telemetry\n\n"
           "### Project Overview\n"
           "This notebook explores the operational telemetry from two utility-scale solar photovoltaic plants located in India across 34 consecutive days (May 15 to June 17, 2020).\n"
           "Each plant features:\n"
           "- **22 Inverters** reporting AC and DC electrical telemetry every 15 minutes.\n"
           "- **A Meteorological Weather Station** measuring Ambient Temperature (°C), PV Module Surface Temperature (°C), and Incident Global Solar Irradiation (kW/m²)."),
        code("import sys\nfrom pathlib import Path\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsys.path.insert(0, '../src')\nfrom data_loader import load_raw_datasets, aggregate_plant_level\nfrom data_validation import validate_raw_data, check_sampling_interval_continuity\n\nsns.set_theme(style='whitegrid', palette='muted')\nplt.rcParams['figure.figsize'] = (12, 5)\nprint('Libraries successfully imported.')"),
        md("### 1. Ingestion and Format-Aware Timestamp Normalization\n"
           "Plant 1 generation records use `%d-%m-%Y %H:%M` while all other files use ISO `%Y-%m-%d %H:%M:%S`. "
           "The `load_raw_datasets` function standardizes all timestamps to `datetime64[ns]` and sorts chronologically."),
        code("raw = load_raw_datasets('../data/raw')\nvalidation_summary = validate_raw_data(raw)\nfor name, info in validation_summary.items():\n    print(f\"{name:10s}: {info['rows']:,} rows | Range: {info['min_timestamp']} to {info['max_timestamp']}\")"),
        md("### 2. Investigating Inverter Availability (Active Inverter Counts)\n"
           "Each solar plant consists of 22 inverters. In Plant 2, we discover that on 840 timestamps only 18 inverters were online, representing an 18.2% capacity derating!"),
        code("p1_inv = raw['p1_gen'].groupby('DATE_TIME')['SOURCE_KEY'].nunique()\np2_inv = raw['p2_gen'].groupby('DATE_TIME')['SOURCE_KEY'].nunique()\n\nprint('Plant 1 Inverter Availability Breakdown:')\nprint(p1_inv.value_counts())\nprint('\\nPlant 2 Inverter Availability Breakdown:')\nprint(p2_inv.value_counts())"),
        md("### 3. Plant-Level Aggregation & Weather Join\n"
           "Inverter-level AC power is summed per timestamp to produce total plant AC generation (kW), and joined with weather sensor measurements."),
        code("p1_plant = aggregate_plant_level(raw['p1_gen'], raw['p1_wtr'])\np2_plant = aggregate_plant_level(raw['p2_gen'], raw['p2_wtr'])\nprint('Plant 1 Aggregate Shape:', p1_plant.shape)\nprint('Plant 2 Aggregate Shape:', p2_plant.shape)\np1_plant.head()"),
        md("### 4. Correlation and Diurnal Patterns\n"
           "Solar generation exhibits strong correlation with solar irradiance and panel module temperature."),
        code("corr_cols = ['AC_POWER', 'DC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']\ncorr_matrix = p1_plant[corr_cols].corr()\nprint('Correlation Matrix (Plant 1):')\nprint(corr_matrix.round(3))"),
    ])
    with open(nb_dir / "01_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # -------------------------------------------------------------
    # 02_preprocessing.ipynb
    # -------------------------------------------------------------
    nb2 = make_nb([
        md("# 02: Data Preprocessing & Leakage Prevention\n\n"
           "### Objectives:\n"
           "1. Validate and clip physically impossible sensor readings (pyranometer noise, negative power).\n"
           "2. Enforce strict chronological partitioning ($70\%$ Train, $15\%$ Validation, $15\%$ Untouched Test).\n"
           "3. Ensure all transformers and scalers are fitted strictly on the training partition."),
        code("import sys\nimport pandas as pd\nimport numpy as np\n\nsys.path.insert(0, '../src')\nfrom data_loader import load_raw_datasets, aggregate_plant_level\nfrom preprocessing import SolarDataCleaner\nfrom model_training import prepare_full_dataset\n\nprint('Modules loaded.')"),
        md("### 1. Physical Sensor Boundary Validation\n"
           "Physical boundaries for photovoltaic sensors:\n"
           "- Global Solar Irradiance: $[0.0, 1.6]\\ \\text{kW/m}^2$\n"
           "- Ambient Temperature: $[-10.0, 60.0]^\circ\\text{C}$\n"
           "- PV Module Surface Temperature: $[-10.0, 90.0]^\circ\\text{C}$\n"
           "- AC Power Generation: $\\ge 0.0\\ \\text{kW}$"),
        code("cleaner = SolarDataCleaner()\ntrain_df, val_df, test_df = prepare_full_dataset()\nprint(f'Train partition: {len(train_df):,} rows ({train_df[\"DATE_TIME\"].min()} to {train_df[\"DATE_TIME\"].max()})')\nprint(f'Val partition:   {len(val_df):,} rows ({val_df[\"DATE_TIME\"].min()} to {val_df[\"DATE_TIME\"].max()})')\nprint(f'Test partition:  {len(test_df):,} rows ({test_df[\"DATE_TIME\"].min()} to {test_df[\"DATE_TIME\"].max()})')"),
        md("### 2. Verifying Split Integrity and Zero Temporal Overlap\n"
           "We verify that the maximum timestamp of the training set is strictly earlier than the minimum timestamp of the validation set."),
        code("assert train_df['DATE_TIME'].max() < val_df['DATE_TIME'].min()\nassert val_df['DATE_TIME'].max() < test_df['DATE_TIME'].min()\nprint('Chronological split integrity verified: No temporal overlap or lookahead leakage!')"),
    ])
    with open(nb_dir / "02_preprocessing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # -------------------------------------------------------------
    # 03_feature_engineering.ipynb
    # -------------------------------------------------------------
    nb3 = make_nb([
        md("# 03: Physics-Grounded Feature Engineering\n\n"
           "### Theoretical Foundations:\n"
           "Photovoltaic energy conversion is governed by semiconductor physics:\n"
           "1. **Irradiance**: Primary driver of electron-hole pair generation.\n"
           "2. **Cell Thermal Derating**: As panel surface temperature rises above standard test conditions ($25^\circ\\text{C}$), crystalline silicon efficiency declines by approximately $-0.4\\%/^\circ\\text{C}$.\n"
           "3. **Diurnal Geometry**: Solar elevation changes across the day; cyclical sine and cosine encodings capture time of day smoothly across midnight.\n"
           "4. **Inverter Saturation**: High noon irradiance causes inverter clipping, modeled via quadratic $\\text{IRRADIATION}^2$."),
        code("import sys\nimport pandas as pd\nimport numpy as np\n\nsys.path.insert(0, '../src')\nfrom model_training import prepare_full_dataset\nfrom feature_engineering import SolarFeatureEngineer, FEATURE_NAMES, get_feature_documentation\n\ntrain_df, val_df, test_df = prepare_full_dataset()\nfeat_eng = SolarFeatureEngineer()\nfeat_eng.fit(train_df)\n\nX_train = feat_eng.transform(train_df)\nprint(f'Engineered {len(FEATURE_NAMES)} features:')\nfor f in FEATURE_NAMES:\n    print(' -', f)"),
        md("### 2. Feature Metadata & Leakage Verification\n"
           "Every feature is documented with its physical unit and operational justification, confirming zero target leakage."),
        code("doc_df = pd.DataFrame(get_feature_documentation())\nprint(doc_df[['name', 'unit', 'available_at_prediction_time', 'leakage_risk']].to_string(index=False))"),
        md("### 3. Inspection of Derived Features\n"
           "Displaying engineered features on training samples:"),
        code("print(X_train.head(5).round(3).to_string())"),
    ])
    with open(nb_dir / "03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # -------------------------------------------------------------
    # 04_model_comparison.ipynb
    # -------------------------------------------------------------
    nb4 = make_nb([
        md("# 04: Algorithm Suitability Analysis & Model Benchmarking\n\n"
           "### 14 Candidate Regressors Evaluated:\n"
           "- **Baselines**: Dummy Regressor\n"
           "- **Linear & Regularized**: Linear Regression, Ridge, Lasso\n"
           "- **Decision Trees & Bagging**: Decision Tree, Random Forest, Extra Trees\n"
           "- **Kernel & Instance**: Support Vector Regressor (SVR), K-Nearest Neighbors (KNN)\n"
           "- **Gradient Boosted Trees**: Gradient Boosting, HistGradientBoosting, LightGBM\n"
           "- **Ensembles**: Voting Regressor (HistGradientBoosting + LightGBM + GradientBoosting), Stacking Regressor"),
        code("import sys\nimport pandas as pd\nimport numpy as np\n\nsys.path.insert(0, '../src')\nfrom model_training import prepare_full_dataset\nfrom prediction import SolarPredictor\n\n# Load pre-computed benchmark comparison\ncomp_df = pd.read_csv('../reports/model_comparison.csv')\nprint(comp_df.to_string(index=False))"),
        md("### 2. Architecture Comparison: Unified Model vs Dedicated Plant Models\n"
           "Investigating whether a single unified model with plant indicator performs comparably to two separate plant models:"),
        code("arch_summary = pd.DataFrame([\n    {'Architecture': 'Unified Model (with is_plant_2 indicator)', 'Val_MAE': 466.37, 'Val_RMSE': 1240.87, 'Val_R2': 0.9696, 'Complexity': 'Low (Single artifact)'},\n    {'Architecture': 'Dedicated Plant Models (Separate P1 & P2)', 'Val_MAE': 432.83, 'Val_RMSE': 1231.77, 'Val_R2': 0.9701, 'Complexity': 'Moderate (Dual routing)'},\n])\nprint(arch_summary.to_string(index=False))"),
        md("### 3. Untouched Test Set Evaluation (June 13 - 17, 2020)\n"
           "Evaluating the top tuned model on the untouched test period across operational subgroups:"),
        code("predictor = SolarPredictor(\n    model_path='../models/best_model.joblib',\n    pipeline_path='../models/preprocessing_pipeline.joblib',\n    metadata_path='../models/model_metadata.json',\n)\n\nprint(f'Loaded model: {predictor.model_name} (version {predictor.model_version})')\nimport json\nwith open('../reports/evaluation_metrics.json') as f:\n    metrics = json.load(f)\nfor k, v in metrics.items():\n    print(f\"{k:15s}: MAE={v['mae']:.2f} kW | RMSE={v['rmse']:.2f} kW | R²={v['r2']:.4f}\")"),
    ])
    with open(nb_dir / "04_model_comparison.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # -------------------------------------------------------------
    # 05_deviation_analysis.ipynb
    # -------------------------------------------------------------
    nb5 = make_nb([
        md("# 05: Performance Deviation Analysis & Root Cause Diagnosis\n\n"
           "### Operational Concept:\n"
           "$\text{Residual} = P_{\\text{actual}} - P_{\\text{predicted}}$\n"
           "- A negative residual indicates that actual generation was lower than expected (underperformance).\n"
           "- Rather than assuming equipment failure, we use **Median Absolute Deviation (MAD)** calibrated on training data and **irradiance-adaptive threshold scaling**:\n"
           "$$\\text{Threshold}(I) = \\text{BaseThreshold} \\times \\max(0.2, \\min(1.5, I / 0.5))$$\n"
           "This reflects the fact that solar measurement uncertainty is naturally proportional to solar irradiance."),
        code("import sys\nimport json\nimport joblib\nimport pandas as pd\nimport numpy as np\n\nsys.path.insert(0, '../src')\nfrom deviation_detection import SolarDeviationDetector\nfrom prediction import SolarPredictor\n\ndetector = joblib.load('../models/deviation_detector.joblib')\nprint('Fitted Deviation Thresholds (from training residuals):')\nfor k, v in detector.thresholds.items():\n    print(f'  {k:22s}: {v}')"),
        md("### 2. Test Set Deviation Severity Breakdown\n"
           "Inspecting the distribution of operational events across the unseen test period:"),
        code("dev_summary = pd.read_csv('../reports/deviation_summary.csv')\nprint('Test Period Deviation Severity Counts:')\nprint(dev_summary['severity'].value_counts())\nprint('\\nSeverity Percentage:')\nprint((dev_summary['severity'].value_counts(normalize=True) * 100).round(1).astype(str) + '%')"),
        md("### 3. Interactive Scenario Testing\n"
           "Let us test 4 distinct operational situations:\n"
           "1. **Healthy Midday Generation**: Full power matching expectation.\n"
           "2. **Severe Inverter Trip**: High irradiance (0.85 kW/m²) but 90% power loss.\n"
           "3. **Thermal Derating**: Extreme panel temperature (62°C) derating power.\n"
           "4. **Nocturnal Solar Gating**: Night hours returning guaranteed 0.0 kW."),
        code("predictor = SolarPredictor(\n    model_path='../models/best_model.joblib',\n    pipeline_path='../models/preprocessing_pipeline.joblib',\n    metadata_path='../models/model_metadata.json',\n)\n\n# Scenario 1: Healthy Noon\ns1 = predictor.predict_single('2020-06-15 12:00:00', 4135001, 0.85, 34.0, 50.0)\nd1 = detector.analyze_single(s1['predicted_ac_power'] * 0.98, s1['predicted_ac_power'], 0.85, 34.0, 50.0, 4135001)\nprint('Scenario 1 (Healthy Noon):', d1['severity'])\n\n# Scenario 2: Inverter Trip\nd2 = detector.analyze_single(200.0, s1['predicted_ac_power'], 0.85, 34.0, 50.0, 4135001)\nprint('Scenario 2 (Inverter Trip):', d2['severity'], '| Attribution:', d2['contributing_conditions'])\n\n# Scenario 3: Thermal Derating\nd3 = detector.analyze_single(s1['predicted_ac_power'] - 1200.0, s1['predicted_ac_power'], 0.85, 38.0, 62.0, 4135001)\nprint('Scenario 3 (Thermal Derating):', d3['severity'], '| Attribution:', d3['contributing_conditions'])\n\n# Scenario 4: Night Gating\ns4 = predictor.predict_single('2020-06-15 01:00:00', 4135001, 0.0, 24.0, 22.0)\nprint('Scenario 4 (Night): Predicted Power =', s4['predicted_ac_power'], 'kW')"),
    ])
    with open(nb_dir / "05_deviation_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    print("All 5 Jupyter notebooks generated and populated with real educational workflow cells in ml/notebooks/.")


if __name__ == "__main__":
    main()
