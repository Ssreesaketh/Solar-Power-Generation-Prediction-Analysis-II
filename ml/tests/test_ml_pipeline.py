"""
Unit and integration tests for ML pipeline components.
Tests cleaning, feature engineering, physical solar gating, capacity derating,
deviation detection, and performance evaluation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import SolarDataCleaner
from feature_engineering import SolarFeatureEngineer, FEATURE_NAMES
from deviation_detection import SolarDeviationDetector
from evaluation import compute_metrics, compute_subgroup_metrics
from prediction import SolarPredictor


def test_cleaner_clipping():
    cleaner = SolarDataCleaner()
    df = pd.DataFrame({
        "IRRADIATION": [-0.5, 2.5, 0.8],
        "AMBIENT_TEMPERATURE": [-20.0, 75.0, 30.0],
        "MODULE_TEMPERATURE": [-25.0, 110.0, 45.0],
    })
    cleaned = cleaner.transform(df)
    assert cleaned["IRRADIATION"].min() >= 0.0
    assert cleaned["IRRADIATION"].max() <= 1.6
    assert cleaned["AMBIENT_TEMPERATURE"].min() >= -10.0
    assert cleaned["AMBIENT_TEMPERATURE"].max() <= 60.0
    assert cleaned["MODULE_TEMPERATURE"].min() >= -10.0
    assert cleaned["MODULE_TEMPERATURE"].max() <= 90.0


def test_feature_engineering_columns():
    feat_eng = SolarFeatureEngineer()
    df = pd.DataFrame({
        "DATE_TIME": ["2020-05-15 12:00:00", "2020-05-15 00:00:00"],
        "PLANT_ID": [4135001, 4136001],
        "IRRADIATION": [0.75, 0.0],
        "AMBIENT_TEMPERATURE": [32.0, 24.0],
        "MODULE_TEMPERATURE": [48.0, 22.0],
    })
    X = feat_eng.transform(df)
    assert len(X.columns) == len(FEATURE_NAMES)
    for col in FEATURE_NAMES:
        assert col in X.columns
    assert len(X) == 2
    assert X["temp_diff"].iloc[0] == 16.0
    assert X["is_plant_2"].iloc[0] == 0
    assert X["is_plant_2"].iloc[1] == 1
    assert "effective_irradiance" in X.columns
    assert "solar_elevation_proxy" in X.columns


def test_deviation_detection_logic():
    detector = SolarDeviationDetector(baseline_mad=200.0)
    y_true = np.array([500.0, 1000.0, 1500.0, 0.0])
    y_pred = np.array([510.0, 990.0, 1480.0, 0.0])
    irr = np.array([0.4, 0.8, 1.0, 0.0])
    detector.fit(y_true, y_pred, irr)

    # Test normal condition
    res_normal = detector.analyze_single(
        actual_ac_power=1000.0,
        predicted_ac_power=1010.0,
        irradiation=0.8,
        ambient_temperature=30.0,
        module_temperature=45.0,
        plant_id=4135001,
    )
    assert res_normal["severity"] == "Normal"

    # Test extreme deficit condition
    res_extreme = detector.analyze_single(
        actual_ac_power=10.0,
        predicted_ac_power=1000.0,
        irradiation=0.8,
        ambient_temperature=30.0,
        module_temperature=45.0,
        plant_id=4135001,
    )
    assert res_extreme["severity"] in ["Significant Deviation", "Extreme Deviation"]
    assert len(res_extreme["contributing_conditions"]) > 0


def test_solar_predictor_inference():
    predictor = SolarPredictor()
    result = predictor.predict_single(
        timestamp="2020-05-15 12:00:00",
        plant_id=4135001,
        irradiation=0.8,
        ambient_temperature=30.0,
        module_temperature=45.0,
    )
    assert "predicted_ac_power" in result
    assert result["predicted_ac_power"] > 0.0
    assert result["status"] == "SUCCESS"

    # Test night-time solar gating
    result_night = predictor.predict_single(
        timestamp="2020-05-15 01:00:00",
        plant_id=4135001,
        irradiation=0.0,
        ambient_temperature=22.0,
        module_temperature=21.0,
    )
    assert result_night["predicted_ac_power"] == 0.0


def test_solar_predictor_capacity_derating():
    predictor = SolarPredictor()
    full_pred = predictor.predict_single(
        timestamp="2020-05-15 12:00:00",
        plant_id=4135001,
        irradiation=0.8,
        ambient_temperature=30.0,
        module_temperature=45.0,
        active_inverters=22,
    )
    derated_pred = predictor.predict_single(
        timestamp="2020-05-15 12:00:00",
        plant_id=4135001,
        irradiation=0.8,
        ambient_temperature=30.0,
        module_temperature=45.0,
        active_inverters=11,  # 50% capacity
    )
    assert round(derated_pred["predicted_ac_power"] / full_pred["predicted_ac_power"], 1) == 0.5
    assert derated_pred["derating_applied"] is True


def test_metrics_computation():
    y_t = np.array([100.0, 200.0, 300.0])
    y_p = np.array([105.0, 195.0, 300.0])
    m = compute_metrics(y_t, y_p)
    assert m["mae"] == round(10.0 / 3.0, 4)
    assert m["r2"] > 0.99
