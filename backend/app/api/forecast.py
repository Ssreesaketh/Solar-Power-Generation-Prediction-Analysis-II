"""15-minute plant-level AC_POWER forecast API using the trained SDC-1 artifact."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Reuse the exact feature engineering and inference code used to train the artifact.
from model import HISTORY_LENGTH, load_artifact, predict_from_history

router = APIRouter(tags=["15-minute Plant Forecast"])
SUPPORTED_PLANTS = (4135001, 4136001)

# backend/model.pkl is committed beside backend/model.py in this repository.
MODEL_PATH = Path(os.getenv("FORECAST_MODEL_PATH", str(Path(__file__).resolve().parents[2] / "model.pkl")))
try:
    FORECAST_ARTIFACT = load_artifact(MODEL_PATH)
except Exception as exc:
    FORECAST_ARTIFACT = None
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


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


@router.get("/forecast-health")
def forecast_health():
    return {
        "model_loaded": FORECAST_ARTIFACT is not None,
        "model_path": str(MODEL_PATH),
        "model_load_error": MODEL_LOAD_ERROR,
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
            detail=f"15-minute forecast model could not be loaded from {MODEL_PATH}. "
                   f"Model loading error: {MODEL_LOAD_ERROR}",
        )
    try:
        records = [reading.model_dump() for reading in request.history]
        return predict_from_history(records, FORECAST_ARTIFACT)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {exc}") from exc
