"""
Data loading module for Solar Power Generation and Weather Sensor datasets.
Handles file ingestion, format-aware timestamp parsing, and basic schema validation.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd


def load_raw_datasets(data_dir: str = "ml/data/raw") -> Dict[str, pd.DataFrame]:
    """
    Loads all four raw solar datasets from the given directory.
    Standardizes timestamps across plants:
      - Plant 1 Generation uses %d-%m-%Y %H:%M
      - Plant 1 Weather uses %Y-%m-%d %H:%M:%S
      - Plant 2 Generation uses %Y-%m-%d %H:%M:%S
      - Plant 2 Weather uses %Y-%m-%d %H:%M:%S
    """
    data_path = Path(data_dir)
    p1_gen_path = data_path / "Plant_1_Generation_Data.csv"
    p1_wtr_path = data_path / "Plant_1_Weather_Sensor_Data.csv"
    p2_gen_path = data_path / "Plant_2_Generation_Data.csv"
    p2_wtr_path = data_path / "Plant_2_Weather_Sensor_Data.csv"

    for p in [p1_gen_path, p1_wtr_path, p2_gen_path, p2_wtr_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required raw dataset file not found: {p}")

    p1_gen = pd.read_csv(p1_gen_path)
    p1_wtr = pd.read_csv(p1_wtr_path)
    p2_gen = pd.read_csv(p2_gen_path)
    p2_wtr = pd.read_csv(p2_wtr_path)

    # Standardize DATE_TIME to datetime64[ns]
    p1_gen["DATE_TIME"] = pd.to_datetime(p1_gen["DATE_TIME"], format="%d-%m-%Y %H:%M")
    p1_wtr["DATE_TIME"] = pd.to_datetime(p1_wtr["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    p2_gen["DATE_TIME"] = pd.to_datetime(p2_gen["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")
    p2_wtr["DATE_TIME"] = pd.to_datetime(p2_wtr["DATE_TIME"], format="%Y-%m-%d %H:%M:%S")

    # Sort each chronologically
    p1_gen = p1_gen.sort_values(by=["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)
    p1_wtr = p1_wtr.sort_values(by="DATE_TIME").reset_index(drop=True)
    p2_gen = p2_gen.sort_values(by=["DATE_TIME", "SOURCE_KEY"]).reset_index(drop=True)
    p2_wtr = p2_wtr.sort_values(by="DATE_TIME").reset_index(drop=True)

    return {
        "p1_gen": p1_gen,
        "p1_wtr": p1_wtr,
        "p2_gen": p2_gen,
        "p2_wtr": p2_wtr,
    }


def aggregate_plant_level(
    gen_df: pd.DataFrame, wtr_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates inverter generation to plant-level per timestamp and joins with plant weather sensors.
    
    Plant-level generation aggregates:
      - AC_POWER: sum of AC_POWER across all active inverters
      - DC_POWER: sum of DC_POWER across all active inverters
      - INVERTER_COUNT: number of reporting inverters
      - DAILY_YIELD: sum of DAILY_YIELD across inverters
      - TOTAL_YIELD: sum of TOTAL_YIELD across inverters
    
    Joined with Weather sensors:
      - AMBIENT_TEMPERATURE
      - MODULE_TEMPERATURE
      - IRRADIATION
    """
    plant_id = gen_df["PLANT_ID"].iloc[0]

    gen_agg = (
        gen_df.groupby("DATE_TIME")
        .agg(
            PLANT_ID=("PLANT_ID", "first"),
            AC_POWER=("AC_POWER", "sum"),
            DC_POWER=("DC_POWER", "sum"),
            INVERTER_COUNT=("SOURCE_KEY", "nunique"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
            TOTAL_YIELD=("TOTAL_YIELD", "sum"),
        )
        .reset_index()
    )

    # Weather sensor data may have one sensor key per plant
    wtr_agg = (
        wtr_df.groupby("DATE_TIME")
        .agg(
            AMBIENT_TEMPERATURE=("AMBIENT_TEMPERATURE", "mean"),
            MODULE_TEMPERATURE=("MODULE_TEMPERATURE", "mean"),
            IRRADIATION=("IRRADIATION", "mean"),
        )
        .reset_index()
    )

    merged = pd.merge(gen_agg, wtr_agg, on="DATE_TIME", how="inner")
    merged["PLANT_ID"] = plant_id
    merged = merged.sort_values(by="DATE_TIME").reset_index(drop=True)
    return merged
