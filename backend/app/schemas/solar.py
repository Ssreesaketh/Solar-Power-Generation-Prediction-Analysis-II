"""
Pydantic Schemas for Solar Power Generation Prediction, Validation and Deviation Analysis.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EnvironmentalTelemetryBase(BaseModel):
    timestamp: datetime = Field(
        ...,
        description="ISO-8601 formatted timestamp of the observation.",
        examples=["2020-05-15T12:00:00Z"]
    )
    plant_id: int = Field(
        ...,
        description="Target plant identifier (4135001 for Plant 1, 4136001 for Plant 2).",
        examples=[4135001]
    )
    irradiation: float = Field(
        ...,
        ge=0.0,
        le=1.6,
        description="Incident solar irradiation in kW/m² [0.0, 1.6].",
        examples=[0.75]
    )
    ambient_temperature: float = Field(
        ...,
        ge=-10.0,
        le=60.0,
        description="Ambient air temperature in °C [-10.0, 60.0].",
        examples=[32.5]
    )
    module_temperature: float = Field(
        ...,
        ge=-10.0,
        le=90.0,
        description="Photovoltaic module surface temperature in °C [-10.0, 90.0].",
        examples=[48.2]
    )
    active_inverters: Optional[int] = Field(
        default=22,
        ge=0,
        le=22,
        description="Operational reporting inverters (nominal capacity is 22).",
        examples=[22]
    )

    @field_validator("plant_id")
    @classmethod
    def validate_plant_id(cls, v: int) -> int:
        valid_plants = {4135001, 4136001}
        if v not in valid_plants:
            raise ValueError(f"Unsupported plant_id: {v}. Supported plant IDs are {valid_plants}")
        return v


class PredictionRequest(EnvironmentalTelemetryBase):
    pass


class PredictionResponse(BaseModel):
    id: Optional[int] = None
    timestamp: datetime
    plant_id: int
    predicted_ac_power: float = Field(..., description="Estimated plant AC power generation in kW.")
    active_inverters: int = 22
    derating_applied: bool = False
    derating_details: Optional[str] = None
    model_version: str
    status: str = "SUCCESS"


class BatchPredictionRequest(BaseModel):
    records: List[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Batch of up to 500 environmental measurement records."
    )


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_records: int
    model_version: str


class DeviationAnalysisRequest(EnvironmentalTelemetryBase):
    actual_ac_power: float = Field(
        ...,
        ge=0.0,
        description="Actual measured plant AC power generation in kW."
    )


class DeviationAnalysisResponse(BaseModel):
    id: Optional[int] = None
    prediction_id: Optional[int] = None
    timestamp: datetime
    plant_id: int
    actual_ac_power: float
    predicted_ac_power: float
    residual: float = Field(..., description="Actual AC Power - Predicted AC Power (kW).")
    absolute_deviation: float
    deviation_percentage: Optional[float] = Field(None, description="Percentage deviation if predicted power >= 50 kW.")
    severity: str = Field(..., description="Categorized severity: Normal, Moderate Deviation, Significant Deviation, Extreme Deviation.")
    contributing_conditions: List[str]
    investigation_status: str
    active_inverters: int = 22
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    target_variable: str
    training_timestamp: Optional[str] = None
    selected_features: List[str]
    feature_count: int
    validation_metrics: Dict[str, Any]
    model_status: str


class StatisticsResponse(BaseModel):
    total_predictions: int
    total_deviations: int
    plant_prediction_counts: Dict[str, int]
    deviation_severity_counts: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    database_connected: bool
    model_loaded: bool
    active_model_version: str
    timestamp: datetime
