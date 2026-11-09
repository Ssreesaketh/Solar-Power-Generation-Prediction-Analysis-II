"""
Feature Engineering Module for Solar Power Generation.
Builds time-based, cyclical, meteorological, and physics-based interaction features.
Prevents target leakage and maintains strict inference reproducibility.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "IRRADIATION",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "temp_diff",
    "irr_mod_interaction",
    "irr_amb_interaction",
    "irradiation_squared",
    "sin_hour",
    "cos_hour",
    "hour",
    "day_of_week",
    "day_of_year",
    "is_plant_2",
    "effective_irradiance",
    "solar_elevation_proxy",
]


class SolarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw environmental measurements and timestamps into predictive features.
    
    Features engineered:
      1. IRRADIATION: Solar irradiance (kW/m²). Primary physical driver of photovoltaic power.
      2. AMBIENT_TEMPERATURE: Ambient air temp (°C) influencing heat dissipation.
      3. MODULE_TEMPERATURE: PV panel surface temp (°C). Higher temp reduces cell efficiency.
      4. temp_diff: MODULE_TEMPERATURE - AMBIENT_TEMPERATURE. Reflects radiative solar heating.
      5. irr_mod_interaction: IRRADIATION * MODULE_TEMPERATURE. Joint effect of sun & thermal state.
      6. irr_amb_interaction: IRRADIATION * AMBIENT_TEMPERATURE. Ambient thermal interaction.
      7. irradiation_squared: IRRADIATION ** 2. Captures inverter clipping & non-linear saturation.
      8. sin_hour: Cyclical diurnal sine encoding: sin(2 * pi * decimal_hour / 24).
      9. cos_hour: Cyclical diurnal cosine encoding: cos(2 * pi * decimal_hour / 24).
      10. hour: Decimal hour of the day [0.0, 24.0).
      11. day_of_week: Day of week index (0=Monday, 6=Sunday).
      12. day_of_year: Day of year [1, 366] capturing seasonal solar declination.
      13. is_plant_2: Binary indicator for Plant 2 (4136001 vs 4135001).
      14. effective_irradiance: Irradiance adjusted for standard crystalline Si thermal loss (-0.4%/°C above 25°C).
      15. solar_elevation_proxy: Geometric diurnal elevation proxy: sin(pi * (hour - 6) / 12) during daylight.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # 1. Timestamp & Diurnal Features
        if "DATE_TIME" in df.columns:
            dt = pd.to_datetime(df["DATE_TIME"])
            decimal_hour = dt.dt.hour + dt.dt.minute / 60.0
            df["hour"] = decimal_hour
            df["day_of_week"] = dt.dt.dayofweek
            df["day_of_year"] = dt.dt.dayofyear
            df["sin_hour"] = np.sin(2.0 * np.pi * decimal_hour / 24.0)
            df["cos_hour"] = np.cos(2.0 * np.pi * decimal_hour / 24.0)
        else:
            hour_val = df.get("hour", 12.0)
            df["hour"] = hour_val
            df["sin_hour"] = np.sin(2.0 * np.pi * hour_val / 24.0)
            df["cos_hour"] = np.cos(2.0 * np.pi * hour_val / 24.0)
            if "day_of_week" not in df.columns:
                df["day_of_week"] = 0
            if "day_of_year" not in df.columns:
                df["day_of_year"] = 150  # Mid-year fallback (~May/June)

        # 2. Thermal Differentials & Physical Interactions
        df["temp_diff"] = df["MODULE_TEMPERATURE"] - df["AMBIENT_TEMPERATURE"]
        df["irr_mod_interaction"] = df["IRRADIATION"] * df["MODULE_TEMPERATURE"]
        df["irr_amb_interaction"] = df["IRRADIATION"] * df["AMBIENT_TEMPERATURE"]
        df["irradiation_squared"] = df["IRRADIATION"] ** 2

        # 3. Physics-based Photovoltaic Derivations
        # Photovoltaic temperature coefficient derating (-0.4%/°C above 25°C STC standard test conditions)
        thermal_loss = np.maximum(0.0, (df["MODULE_TEMPERATURE"] - 25.0) * 0.004)
        df["effective_irradiance"] = df["IRRADIATION"] * (1.0 - thermal_loss)

        # Solar elevation geometric proxy: peak elevation at solar noon (12:00), zero before 06:00 and after 18:00
        sun_window = np.clip(df["hour"] - 6.0, 0.0, 12.0)
        df["solar_elevation_proxy"] = np.maximum(0.0, np.sin(np.pi * sun_window / 12.0))

        # 4. Plant Indicator
        if "PLANT_ID" in df.columns:
            df["is_plant_2"] = (df["PLANT_ID"] == 4136001).astype(int)
        elif "is_plant_2" not in df.columns:
            df["is_plant_2"] = 0

        # Return aligned DataFrame containing strictly the selected features
        return df[self.feature_names]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        return list(self.feature_names)


def get_feature_documentation() -> List[Dict]:
    """
    Returns full metadata documentation for all engineered features.
    """
    return [
        {
            "name": "IRRADIATION",
            "type": "float",
            "unit": "kW/m²",
            "description": "Global solar irradiance incident on solar panels.",
            "operational_justification": "Primary physical driver of photovoltaic electron flow.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "AMBIENT_TEMPERATURE",
            "type": "float",
            "unit": "°C",
            "description": "Surrounding ambient air temperature measured by meteorological station.",
            "operational_justification": "Influences heat dissipation and ambient operating equilibrium.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "MODULE_TEMPERATURE",
            "type": "float",
            "unit": "°C",
            "description": "Surface temperature of solar photovoltaic panel modules.",
            "operational_justification": "PV panel efficiency decreases linearly as cell temperature rises above 25°C.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "temp_diff",
            "type": "float",
            "unit": "°C",
            "description": "MODULE_TEMPERATURE minus AMBIENT_TEMPERATURE.",
            "operational_justification": "Direct measure of solar absorption heating above ambient.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "irr_mod_interaction",
            "type": "float",
            "unit": "kW·°C/m²",
            "description": "Interaction product of IRRADIATION and MODULE_TEMPERATURE.",
            "operational_justification": "Captures joint non-linear effect of peak sun exposure and thermal degradation.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "irr_amb_interaction",
            "type": "float",
            "unit": "kW·°C/m²",
            "description": "Interaction product of IRRADIATION and AMBIENT_TEMPERATURE.",
            "operational_justification": "Captures seasonal/meteorological climate interaction.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "irradiation_squared",
            "type": "float",
            "unit": "(kW/m²)²",
            "description": "Quadratic transformation of IRRADIATION.",
            "operational_justification": "Models non-linear saturation curve of inverter clipping at high irradiance.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "sin_hour",
            "type": "float",
            "unit": "dimensionless [-1, 1]",
            "description": "Sine cyclical encoding of time of day: sin(2 * pi * hour / 24).",
            "operational_justification": "Provides smooth continuous representation of the 24-hour diurnal cycle.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "cos_hour",
            "type": "float",
            "unit": "dimensionless [-1, 1]",
            "description": "Cosine cyclical encoding of time of day: cos(2 * pi * hour / 24).",
            "operational_justification": "Provides perpendicular cyclical component to sin_hour.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "hour",
            "type": "float",
            "unit": "hours [0, 24)",
            "description": "Continuous decimal hour of day.",
            "operational_justification": "Allows tree splits on morning vs afternoon solar geometry.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "day_of_week",
            "type": "int",
            "unit": "index [0, 6]",
            "description": "Day of week (Monday=0 to Sunday=6).",
            "operational_justification": "Encodes potential weekly dispatch or maintenance rhythms.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "day_of_year",
            "type": "int",
            "unit": "day [1, 366]",
            "description": "Calendar day of year.",
            "operational_justification": "Encodes seasonal solar declination and azimuth shifts.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "is_plant_2",
            "type": "int",
            "unit": "binary [0, 1]",
            "description": "Binary indicator for Plant 2 (4136001).",
            "operational_justification": "Encodes baseline capacity and hardware differences between Plant 1 and Plant 2.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "effective_irradiance",
            "type": "float",
            "unit": "kW/m²",
            "description": "Irradiance adjusted by PV thermal loss coefficient (-0.4%/°C above 25°C).",
            "operational_justification": "Direct physical representation of actual usable solar flux after cell heating.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
        {
            "name": "solar_elevation_proxy",
            "type": "float",
            "unit": "dimensionless [0, 1]",
            "description": "Idealized solar elevation angle proxy based on daytime sine curve.",
            "operational_justification": "Provides clean bell-shaped daylight geometric prior peaking at noon.",
            "available_at_prediction_time": True,
            "leakage_risk": False,
        },
    ]
