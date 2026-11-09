"""
Data validation module.
Verifies data integrity, missing values, valid numerical ranges, and timestamp continuity.
"""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


def validate_raw_data(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    Validates shapes, missing values, unique values, and duplicates in raw datasets.
    """
    report = {}
    for name, df in data_dict.items():
        missing_counts = df.isnull().sum().to_dict()
        dup_count = int(df.duplicated().sum())
        date_col = "DATE_TIME"
        min_date = str(df[date_col].min()) if date_col in df.columns else None
        max_date = str(df[date_col].max()) if date_col in df.columns else None

        report[name] = {
            "rows": len(df),
            "columns": list(df.columns),
            "missing_values": missing_counts,
            "duplicate_rows": dup_count,
            "min_timestamp": min_date,
            "max_timestamp": max_date,
            "unique_timestamps": int(df[date_col].nunique()) if date_col in df.columns else 0,
        }
    return report


def check_sensor_ranges(df: pd.DataFrame) -> List[str]:
    """
    Validates that sensor readings fall within physically plausible physical ranges:
      - AMBIENT_TEMPERATURE: -10 C to 60 C
      - MODULE_TEMPERATURE: -10 C to 90 C
      - IRRADIATION: 0.0 to 1.6 kW/m^2
      - AC_POWER: >= 0.0
    """
    issues = []
    if "IRRADIATION" in df.columns:
        neg_irr = (df["IRRADIATION"] < 0).sum()
        if neg_irr > 0:
            issues.append(f"{neg_irr} negative irradiation records found")
        excess_irr = (df["IRRADIATION"] > 1.6).sum()
        if excess_irr > 0:
            issues.append(f"{excess_irr} irradiation records above 1.6 kW/m^2")

    if "AMBIENT_TEMPERATURE" in df.columns:
        extreme_amb = ((df["AMBIENT_TEMPERATURE"] < -10) | (df["AMBIENT_TEMPERATURE"] > 60)).sum()
        if extreme_amb > 0:
            issues.append(f"{extreme_amb} ambient temperature readings outside -10C..60C")

    if "MODULE_TEMPERATURE" in df.columns:
        extreme_mod = ((df["MODULE_TEMPERATURE"] < -10) | (df["MODULE_TEMPERATURE"] > 90)).sum()
        if extreme_mod > 0:
            issues.append(f"{extreme_mod} module temperature readings outside -10C..90C")

    if "AC_POWER" in df.columns:
        neg_ac = (df["AC_POWER"] < 0).sum()
        if neg_ac > 0:
            issues.append(f"{neg_ac} negative AC_POWER records found")

    return issues


def check_sampling_interval_continuity(df: pd.DataFrame, expected_minutes: int = 15) -> Dict:
    """
    Checks for missing timestamps along the time series.
    """
    df_sorted = df.sort_values(by="DATE_TIME").drop_duplicates(subset=["DATE_TIME"])
    diffs = df_sorted["DATE_TIME"].diff()
    expected_delta = pd.Timedelta(minutes=expected_minutes)
    gaps = diffs[diffs > expected_delta]
    return {
        "total_records": len(df_sorted),
        "expected_interval_min": expected_minutes,
        "gaps_found": len(gaps),
        "max_gap": str(diffs.max()),
        "regular_fraction": float((diffs == expected_delta).mean()) if len(diffs) > 1 else 1.0,
    }
