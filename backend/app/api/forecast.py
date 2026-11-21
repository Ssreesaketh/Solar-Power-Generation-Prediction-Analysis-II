"""15-minute plant-level AC_POWER forecasting endpoints for the SDC-1 RF model."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import PROJECT_ROOT

router = APIRouter(tags=["15-minute Plant Forecast"])
HISTORY_LENGTH = 97
SUPPORTED_PLANTS = (4135001, 4136001)
MODEL_PATH = Path(os.getenv(
    "FORECAST_MODEL_PATH",
    str(PROJECT_ROOT / "ml" / "models" / "plant_forecast_rf.pkl"),
))
try:
    FORECAST_ARTIFACT = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    FORECAST_ARTIFACT = None


class Reading(BaseModel):
    timestamp: datetime
    plant_id: int
    ac_power: Optional[float] = None
    dc_power: Optional[float] = None
    ambient_temperature: Optional[float] = None
    module_temperature: Optional[float] = None
    irradiation: Optional[float] = None
    inverter_count: Optional[float] = 22


class ForecastRequest(BaseModel):
    history: list[Reading] = Field(min_length=HISTORY_LENGTH, max_length=HISTORY_LENGTH)


def _feature_row(history: list[dict], columns: list[str]) -> pd.DataFrame:
    if len(history) != HISTORY_LENGTH:
        raise ValueError(f"Exactly {HISTORY_LENGTH} readings are required.")
    frame = pd.DataFrame(history)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Every timestamp must be a valid date/time.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("History must be ordered oldest to newest.")
    plant_ids = frame["plant_id"].dropna().astype(int).unique()
    if len(plant_ids) != 1 or int(plant_ids[0]) not in SUPPORTED_PLANTS:
        raise ValueError(f"All readings must belong to one supported plant: {SUPPORTED_PLANTS}.")
    if not (frame["timestamp"].diff().dropna() == pd.Timedelta(minutes=15)).all():
        raise ValueError("Readings must be consecutive 15-minute intervals.")

    latest = frame.iloc[-1]
    ts = latest["timestamp"]
    f = {
        "PLANT_ID": int(latest["plant_id"]),
        "AC_POWER": latest["ac_power"], "DC_POWER": latest["dc_power"],
        "AMBIENT_TEMPERATURE": latest["ambient_temperature"],
        "MODULE_TEMPERATURE": latest["module_temperature"],
        "IRRADIATION": latest["irradiation"], "INVERTER_COUNT": latest["inverter_count"],
        "hour": ts.hour, "dayofweek": ts.dayofweek, "dayofyear": ts.dayofyear,
        "hour_sin": np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos": np.cos(2 * np.pi * ts.hour / 24),
        "doy_sin": np.sin(2 * np.pi * ts.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * ts.dayofyear / 365.25),
    }
    lag_pos = {1: -2, 2: -3, 4: -5, 96: -97}
    source = {
        "AC_POWER": "ac_power", "DC_POWER": "dc_power",
        "IRRADIATION": "irradiation", "AMBIENT_TEMPERATURE": "ambient_temperature",
        "MODULE_TEMPERATURE": "module_temperature",
    }
    for col, key in source.items():
        for lag, pos in lag_pos.items():
            f[f"{col}_lag_{lag}"] = frame.iloc[pos][key]
    return pd.DataFrame([[f.get(c, np.nan) for c in columns]], columns=columns)


@router.get("/forecast-health")
def forecast_health():
    return {
        "model_loaded": FORECAST_ARTIFACT is not None,
        "model_path": str(MODEL_PATH),
        "plants_supported": list(SUPPORTED_PLANTS),
        "forecast_horizon_minutes": 15,
    }


@router.get("/sample-request")
def sample_request(plant_id: int = 4135001):
    if plant_id not in SUPPORTED_PLANTS:
        raise HTTPException(status_code=400, detail="plant_id must be 4135001 or 4136001")
    end = datetime(2020, 6, 18, 12, 0)
    history = []
    for i in range(HISTORY_LENGTH):
        ts = end - timedelta(minutes=15 * (HISTORY_LENGTH - 1 - i))
        hour = ts.hour + ts.minute / 60
        daylight = max(0.0, np.sin(np.pi * (hour - 6) / 12))
        scale = 1.0 if plant_id == 4135001 else 0.8
        ac = float(18000 * daylight * scale)
        history.append({
            "timestamp": ts.isoformat(), "plant_id": plant_id,
            "ac_power": ac, "dc_power": ac * 1.06,
            "ambient_temperature": 27.0 + 4.0 * daylight,
            "module_temperature": 28.0 + 18.0 * daylight,
            "irradiation": 0.8 * daylight, "inverter_count": 22,
        })
    return {
        "notice": "Synthetic history for interface testing only; not validated operational telemetry.",
        "history": history,
    }


@router.post("/predict")
def predict(request: ForecastRequest):
    if FORECAST_ARTIFACT is None:
        raise HTTPException(
            status_code=503,
            detail="15-minute forecast model is not installed. Add ml/models/plant_forecast_rf.pkl before deployment.",
        )
    try:
        records = [r.model_dump() for r in request.history]
        feature_columns = FORECAST_ARTIFACT["feature_columns"]
        row = _feature_row(records, feature_columns)
        transformed = FORECAST_ARTIFACT["imputer"].transform(row)
        value = float(FORECAST_ARTIFACT["model"].predict(transformed)[0])
        current = pd.Timestamp(records[-1]["timestamp"])
        return {
            "prediction_level": "plant", "plant_id": int(records[-1]["plant_id"]),
            "target": "AC_POWER", "forecast_horizon_minutes": 15,
            "input_timestamp": current.isoformat(),
            "forecast_timestamp": (current + pd.Timedelta(minutes=15)).isoformat(),
            "predicted_ac_power": value,
            "model": FORECAST_ARTIFACT.get("model_name", "Random Forest"),
            "units": "dataset AC_POWER units",
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc
