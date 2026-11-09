"""
SQLAlchemy ORM Database Models for Solar Power Prediction and Deviation Tracking.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    BigInteger,
    Index,
)
from sqlalchemy.orm import relationship
from app.database.session import Base


class Plant(Base):
    __tablename__ = "plants"

    id = Column(Integer, primary_key=True, index=True)
    external_plant_id = Column(BigInteger, unique=True, nullable=False, index=True)
    plant_name = Column(String(100), nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    predictions = relationship("Prediction", back_populates="plant")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), unique=True, nullable=False, index=True)
    target_variable = Column(String(50), default="AC_POWER")
    training_timestamp = Column(DateTime(timezone=True), nullable=True)
    model_metadata = Column(JSON, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    predictions = relationship("Prediction", back_populates="model_version_rel")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    plant_id = Column(BigInteger, ForeignKey("plants.external_plant_id"), nullable=False, index=True)
    actual_ac_power = Column(Float, nullable=True)
    predicted_ac_power = Column(Float, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False, index=True)
    input_features = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    plant = relationship("Plant", back_populates="predictions")
    model_version_rel = relationship("ModelVersion", back_populates="predictions")
    deviation_event = relationship("DeviationEvent", back_populates="prediction", uselist=False, cascade="all, delete-orphan")


class DeviationEvent(Base):
    __tablename__ = "deviation_events"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    residual = Column(Float, nullable=False)
    absolute_deviation = Column(Float, nullable=False)
    deviation_percentage = Column(Float, nullable=True)
    severity = Column(String(50), nullable=False, index=True)
    contributing_conditions = Column(JSON, nullable=False)
    investigation_status = Column(String(50), default="PENDING_REVIEW", nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)

    prediction = relationship("Prediction", back_populates="deviation_event")


# Composite indexes for querying time ranges and severities
Index("idx_predictions_plant_timestamp", Prediction.plant_id, Prediction.timestamp)
Index("idx_deviation_severity_created", DeviationEvent.severity, DeviationEvent.created_at)
