"""
Production Model Prediction Module.
Provides validated, reproducible inference using saved preprocessor and trained regressor.
Applies physical solar gating to ensure zero night-time drift and supports inverter capacity scaling.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd


class SolarPredictor:
    """
    Unified predictor wrapper that loads the serialized model and pipeline,
    transforms incoming inputs, and generates predictions with solar gating.
    
    Nominal capacity: 22 inverters per plant.
    If active_inverters is provided and < 22, applies proportional capacity derating.
    """

    NOMINAL_INVERTER_COUNT = 22

    def __init__(
        self,
        model_path: Union[str, Path] = "ml/models/best_model.joblib",
        pipeline_path: Union[str, Path] = "ml/models/preprocessing_pipeline.joblib",
        metadata_path: Union[str, Path] = "ml/models/model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.pipeline_path = Path(pipeline_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {self.pipeline_path}")

        self.model = joblib.load(self.model_path)
        self.pipeline = joblib.load(self.pipeline_path)

        self.metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        self.model_name = self.metadata.get("model_name", "Solar_Power_Predictor")
        self.model_version = self.metadata.get("model_version", "v1.0.0")
        self.target_variable = self.metadata.get("target_variable", "AC_POWER")

    def predict_dataframe(self, df: pd.DataFrame, active_inverters: Optional[int] = None) -> np.ndarray:
        """
        Runs batch prediction on a pandas DataFrame with raw columns.
        Applies physical solar gating:
          - If IRRADIATION == 0 or hour in night window, power is 0.0 kW.
        Optionally applies inverter availability derating:
          - If active_inverters is specified and < 22, scales prediction by (active_inverters / 22.0).
        """
        X = self.pipeline.transform(df)
        raw_preds = self.model.predict(X)
        clipped_preds = np.clip(raw_preds, 0.0, None)

        # Apply physical solar gating
        if "IRRADIATION" in df.columns:
            clipped_preds[df["IRRADIATION"] <= 0.0] = 0.0

        if "DATE_TIME" in df.columns:
            dt = pd.to_datetime(df["DATE_TIME"])
            night_mask = (dt.dt.hour < 5) | (dt.dt.hour >= 20)
            clipped_preds[night_mask] = 0.0

        # Inverter capacity scaling if derated
        if active_inverters is not None and 0 <= active_inverters < self.NOMINAL_INVERTER_COUNT:
            scale_factor = active_inverters / float(self.NOMINAL_INVERTER_COUNT)
            clipped_preds = clipped_preds * scale_factor

        return clipped_preds

    def predict_single(
        self,
        timestamp: str,
        plant_id: int,
        irradiation: float,
        ambient_temperature: float,
        module_temperature: float,
        active_inverters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Runs single record prediction.
        """
        record = {
            "DATE_TIME": [timestamp],
            "PLANT_ID": [int(plant_id)],
            "IRRADIATION": [float(irradiation)],
            "AMBIENT_TEMPERATURE": [float(ambient_temperature)],
            "MODULE_TEMPERATURE": [float(module_temperature)],
        }
        df = pd.DataFrame(record)
        pred = float(self.predict_dataframe(df, active_inverters=active_inverters)[0])

        derating_note = None
        if active_inverters is not None and active_inverters < self.NOMINAL_INVERTER_COUNT:
            derating_note = f"Derated for {active_inverters}/{self.NOMINAL_INVERTER_COUNT} active inverters ({active_inverters/self.NOMINAL_INVERTER_COUNT*100:.1f}% nominal capacity)"

        return {
            "predicted_ac_power": round(pred, 2),
            "timestamp": timestamp,
            "plant_id": plant_id,
            "active_inverters": active_inverters or self.NOMINAL_INVERTER_COUNT,
            "derating_applied": derating_note is not None,
            "derating_details": derating_note,
            "model_version": self.model_version,
            "status": "SUCCESS",
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs prediction for a list of input dictionaries.
        """
        df = pd.DataFrame(records)
        df = df.rename(columns={
            "timestamp": "DATE_TIME",
            "plant_id": "PLANT_ID",
            "irradiation": "IRRADIATION",
            "ambient_temperature": "AMBIENT_TEMPERATURE",
            "module_temperature": "MODULE_TEMPERATURE",
        })
        preds = self.predict_dataframe(df)

        output = []
        for i, row in df.iterrows():
            output.append({
                "predicted_ac_power": round(float(preds[i]), 2),
                "timestamp": str(row["DATE_TIME"]),
                "plant_id": int(row["PLANT_ID"]),
                "model_version": self.model_version,
                "status": "SUCCESS",
            })
        return output
