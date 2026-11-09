"""
FastAPI V1 API Endpoints for Predictions, Deviations, Metadata and Query History.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.entities import Prediction, DeviationEvent
from app.schemas.solar import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    DeviationAnalysisRequest,
    DeviationAnalysisResponse,
    ModelInfoResponse,
    StatisticsResponse,
)
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/model/info", response_model=ModelInfoResponse, summary="Retrieve Active Model Manifest & Validation Metrics")
def get_model_info():
    """
    Returns the metadata, feature manifest, algorithm details, and evaluation metrics of the active model.
    """
    if not ml_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not currently loaded.")
    return ml_service.get_model_info()


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_201_CREATED, summary="Single Solar Power Generation Prediction")
def predict_single(
    request: PredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Computes expected AC power generation (kW) from environmental telemetry:
      - Validates input boundaries (irradiation, ambient temp, module temp).
      - Applies physical solar gating to eliminate night-time drift.
      - Stores prediction record in PostgreSQL.
    """
    if not ml_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is currently unavailable.")
    try:
        return ml_service.predict_single(request, db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_201_CREATED, summary="Batch Solar Power Generation Prediction")
def predict_batch(
    batch_request: BatchPredictionRequest,
    db: Session = Depends(get_db),
):
    """
    Performs vector batch prediction for up to 500 environmental records.
    """
    if not ml_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is currently unavailable.")
    try:
        return ml_service.predict_batch(batch_request, db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/deviations/analyze", response_model=DeviationAnalysisResponse, status_code=status.HTTP_201_CREATED, summary="Analyze Generation Deviation and Root Cause")
def analyze_deviation(
    request: DeviationAnalysisRequest,
    db: Session = Depends(get_db),
):
    """
    Compares actual generation against model prediction:
      - Computes residual, absolute deviation, and percentage deviation.
      - Classifies severity: Normal, Moderate Deviation, Significant Deviation, Extreme Deviation.
      - Identifies contributing environmental and operational factors.
      - Persists investigation record in PostgreSQL.
    """
    if not ml_service.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model service is currently unavailable.")
    try:
        return ml_service.analyze_deviation(request, db)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/predictions", summary="Query Prediction History with Pagination and Filters")
def list_predictions(
    plant_id: Annotated[Optional[int], Query(description="Filter by plant ID (4135001 or 4136001)")] = None,
    start_date: Annotated[Optional[datetime], Query(description="Filter start timestamp")] = None,
    end_date: Annotated[Optional[datetime], Query(description="Filter end timestamp")] = None,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Page size")] = 50,
    db: Session = Depends(get_db),
):
    """
    Returns paginated prediction history with optional filtering.
    """
    query = db.query(Prediction)
    if plant_id:
        query = query.filter(Prediction.plant_id == plant_id)
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)

    total = query.count()
    records = query.order_by(Prediction.timestamp.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": [
            {
                "id": r.id,
                "timestamp": r.timestamp,
                "plant_id": r.plant_id,
                "predicted_ac_power": r.predicted_ac_power,
                "actual_ac_power": r.actual_ac_power,
                "input_features": r.input_features,
                "created_at": r.created_at,
            }
            for r in records
        ],
    }


@router.get("/deviations", summary="Query Deviation History with Pagination and Filters")
def list_deviations(
    plant_id: Annotated[Optional[int], Query(description="Filter by plant ID")] = None,
    severity: Annotated[Optional[str], Query(description="Filter by severity (Normal, Moderate Deviation, Significant Deviation, Extreme Deviation)")] = None,
    start_date: Annotated[Optional[datetime], Query(description="Filter start timestamp")] = None,
    end_date: Annotated[Optional[datetime], Query(description="Filter end timestamp")] = None,
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=200, description="Page size")] = 50,
    db: Session = Depends(get_db),
):
    """
    Returns paginated deviation events with severity and root-cause factors.
    """
    query = db.query(DeviationEvent).join(Prediction, DeviationEvent.prediction_id == Prediction.id)
    if plant_id:
        query = query.filter(Prediction.plant_id == plant_id)
    if severity:
        query = query.filter(DeviationEvent.severity == severity)
    if start_date:
        query = query.filter(Prediction.timestamp >= start_date)
    if end_date:
        query = query.filter(Prediction.timestamp <= end_date)

    total = query.count()
    events = query.order_by(DeviationEvent.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "records": [
            {
                "id": d.id,
                "prediction_id": d.prediction_id,
                "timestamp": d.prediction.timestamp if d.prediction else None,
                "plant_id": d.prediction.plant_id if d.prediction else None,
                "residual": d.residual,
                "absolute_deviation": d.absolute_deviation,
                "deviation_percentage": d.deviation_percentage,
                "severity": d.severity,
                "contributing_conditions": d.contributing_conditions,
                "investigation_status": d.investigation_status,
                "created_at": d.created_at,
            }
            for d in events
        ],
    }


@router.get("/statistics", response_model=StatisticsResponse, summary="Retrieve Operational and Database Analytics")
def get_statistics(db: Session = Depends(get_db)):
    """
    Aggregates database telemetry counts, plant-wise distributions, and anomaly severity frequencies.
    """
    return ml_service.get_statistics(db)
