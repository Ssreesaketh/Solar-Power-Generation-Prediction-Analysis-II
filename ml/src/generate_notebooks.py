"""
Utility script to generate Jupyter Notebooks for Phase I deliverables.
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

    # 01_eda.ipynb
    nb1 = make_nb([
        md("# 01: Exploratory Data Analysis\n## Solar Power Generation and Weather Telemetry\nExplores raw telemetry from Plant 1 and Plant 2, investigates data quality, sampling continuity, and sensor distributions."),
        code("import sys\nsys.path.insert(0, '../src')\nfrom data_loader import load_raw_datasets, aggregate_plant_level\nfrom data_validation import validate_raw_data\nraw = load_raw_datasets('../data/raw')\nreport = validate_raw_data(raw)\nprint('Raw Data Overview:', report)"),
        code("p1_plant = aggregate_plant_level(raw['p1_gen'], raw['p1_wtr'])\np2_plant = aggregate_plant_level(raw['p2_gen'], raw['p2_wtr'])\nprint('Plant 1 Aggregate Shape:', p1_plant.shape)\nprint('Plant 2 Aggregate Shape:', p2_plant.shape)"),
        code("from run_eda import run\nrun()"),
    ])
    with open(nb_dir / "01_eda.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=2)

    # 02_preprocessing.ipynb
    nb2 = make_nb([
        md("# 02: Data Preprocessing\nCleans sensor anomalies, clips physically invalid readings, aligns generation and weather streams."),
        code("import sys\nsys.path.insert(0, '../src')\nfrom data_loader import load_raw_datasets\nfrom preprocessing import SolarDataCleaner, prepare_dataset\nraw = load_raw_datasets('../data/raw')\ncleaner = SolarDataCleaner()\np1_clean = cleaner.transform(raw['p1_wtr'])\nprint('Preprocessed weather sample:\n', p1_clean.head())"),
    ])
    with open(nb_dir / "02_preprocessing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=2)

    # 03_feature_engineering.ipynb
    nb3 = make_nb([
        md("# 03: Feature Engineering\nConstructs cyclical diurnal features, solar thermal differential, and non-linear irradiance transformations without target leakage."),
        code("import sys\nsys.path.insert(0, '../src')\nfrom model_training import prepare_full_dataset\nfrom feature_engineering import SolarFeatureEngineer, get_feature_documentation\ntrain_df, val_df, test_df = prepare_full_dataset()\nfeat_eng = SolarFeatureEngineer()\nfeat_eng.fit(train_df)\nX_train = feat_eng.transform(train_df)\nprint('Engineered Features:', list(X_train.columns))\nprint(X_train.head())"),
    ])
    with open(nb_dir / "03_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=2)

    # 04_model_comparison.ipynb
    nb4 = make_nb([
        md("# 04: Algorithm Suitability Analysis & Benchmarking\nEvaluates 12 candidate regressors using strict chronological train/validation partitions."),
        code("import sys\nsys.path.insert(0, '../src')\nimport pandas as pd\ncomp_df = pd.read_csv('../reports/model_comparison.csv')\nprint(comp_df.to_string(index=False))"),
    ])
    with open(nb_dir / "04_model_comparison.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=2)

    # 05_deviation_analysis.ipynb
    nb5 = make_nb([
        md("# 05: Deviation Detection and Explainability\nApplies context-aware Median Absolute Deviation thresholds and generates root cause diagnostic attributions."),
        code("import sys\nsys.path.insert(0, '../src')\nimport joblib, pandas as pd\nfrom deviation_detection import SolarDeviationDetector\ndetector = joblib.load('../models/deviation_detector.joblib')\nprint('Fitted Deviation Thresholds:', detector.thresholds)\ndev_df = pd.read_csv('../reports/deviation_summary.csv')\nprint('Test Deviation Breakdown:\n', dev_df['severity'].value_counts())"),
    ])
    with open(nb_dir / "05_deviation_analysis.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=2)

    print("All 5 Jupyter notebooks generated successfully in ml/notebooks/.")


if __name__ == "__main__":
    main()
