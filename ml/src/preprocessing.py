"""
Data Preprocessing Pipeline Module.
Implements scikit-learn compatible transformers for time series solar data.
Fitted strictly on training partitions to prevent data leakage.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class SolarDataCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw solar telemetry records:
      - Validates and clips physically impossible sensor readings.
      - Fills or interpolates small gaps if required.
      - Normalizes column names and types.
    """

    def __init__(
        self,
        ambient_temp_range: Tuple[float, float] = (-10.0, 60.0),
        module_temp_range: Tuple[float, float] = (-10.0, 90.0),
        irradiation_range: Tuple[float, float] = (0.0, 1.6),
    ):
        self.ambient_temp_range = ambient_temp_range
        self.module_temp_range = module_temp_range
        self.irradiation_range = irradiation_range

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "IRRADIATION" in df.columns:
            df["IRRADIATION"] = df["IRRADIATION"].clip(
                lower=self.irradiation_range[0], upper=self.irradiation_range[1]
            )
        if "AMBIENT_TEMPERATURE" in df.columns:
            df["AMBIENT_TEMPERATURE"] = df["AMBIENT_TEMPERATURE"].clip(
                lower=self.ambient_temp_range[0], upper=self.ambient_temp_range[1]
            )
        if "MODULE_TEMPERATURE" in df.columns:
            df["MODULE_TEMPERATURE"] = df["MODULE_TEMPERATURE"].clip(
                lower=self.module_temp_range[0], upper=self.module_temp_range[1]
            )
        return df


def prepare_dataset(
    gen_df: pd.DataFrame,
    wtr_df: pd.DataFrame,
    granularity: str = "plant",
) -> pd.DataFrame:
    """
    Aligns and prepares solar generation and weather telemetry.
    Supports 'plant' level aggregation and 'inverter' level analysis.
    """
    if granularity == "plant":
        from data_loader import aggregate_plant_level
        df = aggregate_plant_level(gen_df, wtr_df)
    else:
        # Inverter-level alignment
        wtr_agg = (
            wtr_df.groupby("DATE_TIME")
            .agg(
                AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
                MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
                IRRADIATION=("IRRADIATION", "mean"),
            )
            .reset_index()
        )
        df = pd.merge(gen_df, wtr_agg, on="DATE_TIME", how="inner")

    df = df.sort_values(by="DATE_TIME").reset_index(drop=True)
    return df
