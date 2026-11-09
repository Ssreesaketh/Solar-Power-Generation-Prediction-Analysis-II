"""
Machine Learning & Deviation Inference Service.
Integrates Phase I SolarPredictor and SolarDeviationDetector with database persistence.
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure ml/src is in sys.path so joblib can deserialize custom transformers
from app.core.config import settings, PROJECT_ROOT
ml_src_path = str((PROJECT_ROOT / "ml" / "src").resolve())
if ml_src_path not in sys.path:
    sys.path.insert(0, ml_src_path)

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.entities import Plant, ModelVersion, Prediction, DeviationEvent
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
from prediction import SolarPredictor
from deviation_detection import SolarDeviationDetector

logger = logging.getLogger("uvicorn.error")


class MLInferenceService:
    def __init__(self):
        self.predictor: Optional[SolarPredictor] = None
        self.detector: Optional[SolarDeviationDetector] = None
        self.metadata: Dict[str, Any] = {}
        self.model_version_id: Optional[int] = None
        self._is_loaded = False

    def load_artifacts(self, db_session: Optional[Session] = None):
        """
        Loads trained Phase I predictor, preprocessor, and deviation detector.
        """
        try:
            model_path = Path(settings.MODEL_PATH)
            pipeline_path = Path(settings.PIPELINE_PATH)
            detector_path = Path(settings.DETECTOR_PATH)
            metadata_path = Path(settings.METADATA_PATH)

            if not model_path.exists() or not pipeline_path.exists():
                logger.error(f"Model artifacts not found at {model_path} / {pipeline_path}")
                self._is_loaded = False
                return

            self.predictor = SolarPredictor(
                model_path=model_path,
                pipeline_path=pipeline_path,
                metadata_path=metadata_path,
            )

            if detector_path.exists():
                self.detector = joblib.load(detector_path)
            else:
                self.detector = SolarDeviationDetector()

            self.metadata = self.predictor.metadata
            self._is_loaded = True
            logger.info(f"Loaded ML model: {self.metadata.get('model_name', 'Solar Model')} version {self.metadata.get('model_version', 'v1.1.0')}")

            if db_session:
                self.sync_db_metadata(db_session)

        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}", exc_info=True)
            self._is_loaded = False

    def sync_db_metadata(self, db: Session):
        """
        Ensures plants and model version exist in the database.
        """
        try:
            plants_seed = [
                {"external_plant_id": 4135001, "plant_name": "Plant 1 Solar Field", "metadata_json": {"inverters": 22, "location": "India", "dc_ac_ratio": 10.0}},
                {"external_plant_id": 4136001, "plant_name": "Plant 2 Solar Field", "metadata_json": {"inverters": 22, "location": "India", "dc_ac_ratio": 1.0}},
            ]
            for p_info in plants_seed:
                existing_p = db.query(Plant).filter(Plant.external_plant_id == p_info["external_plant_id"]).first()
                if not existing_p:
                    p = Plant(**p_info)
                    db.add(p)
            db.commit()

            version_str = self.metadata.get("model_version", "v1.1.0")
            mv = db.query(ModelVersion).filter(ModelVersion.model_version == version_str).first()
            if not mv:
                mv = ModelVersion(
                    model_name=self.metadata.get("model_name", "Solar_Power_Predictor"),
                    model_version=version_str,
                    target_variable=self.metadata.get("target_variable", "AC_POWER"),
                    training_timestamp=datetime.now(timezone.utc),
                    model_metadata=self.metadata,
                    evaluation_metrics=self.metadata.get("test_metrics", {}),
                    is_active=True,
                )
                db.add(mv)
                db.commit()
                db.refresh(mv)
            self.model_version_id = mv.id
        except Exception as e:
            logger.warning(f"Database metadata sync issue: {e}")
            db.rollback()

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def predict_single(self, request: PredictionRequest, db: Session) -> PredictionResponse:
        if not self._is_loaded or self.predictor is None:
            raise RuntimeError("ML model is not loaded.")

        pred_res = self.predictor.predict_single(
            timestamp=str(request.timestamp),
            plant_id=request.plant_id,
            irradiation=request.irradiation,
            ambient_temperature=request.ambient_temperature,
            module_temperature=request.module_temperature,
            active_inverters=request.active_inverters,
        )

        version_str = self.metadata.get("model_version", "v1.1.0")
        if not self.model_version_id:
            self.sync_db_metadata(db)

        # Persist prediction
        pred_record = Prediction(
            timestamp=request.timestamp,
            plant_id=request.plant_id,
            predicted_ac_power=pred_res["predicted_ac_power"],
            model_version_id=self.model_version_id or 1,
            input_features={
                "irradiation": request.irradiation,
                "ambient_temperature": request.ambient_temperature,
                "module_temperature": request.module_temperature,
                "active_inverters": request.active_inverters,
            },
        )
        db.add(pred_record)
        db.commit()
        db.refresh(pred_record)

        return PredictionResponse(
            id=pred_record.id,
            timestamp=request.timestamp,
            plant_id=request.plant_id,
            predicted_ac_power=pred_res["predicted_ac_power"],
            active_inverters=pred_res["active_inverters"],
            derating_applied=pred_res["derating_applied"],
            derating_details=pred_res["derating_details"],
            model_version=version_str,
            status="SUCCESS",
        )

    def predict_batch(self, batch_request: BatchPredictionRequest, db: Session) -> BatchPredictionResponse:
        if not self._is_loaded or self.predictor is None:
            raise RuntimeError("ML model is not loaded.")

        records = [r.model_dump() for r in batch_request.records]
        preds = self.predictor.predict_batch(records)

        version_str = self.metadata.get("model_version", "v1.1.0")
        if not self.model_version_id:
            self.sync_db_metadata(db)

        response_list = []
        db_records = []
        for orig, pred_out in zip(batch_request.records, preds):
            pred_val = pred_out["predicted_ac_power"]
            db_records.append(Prediction(
                timestamp=orig.timestamp,
                plant_id=orig.plant_id,
                predicted_ac_power=pred_val,
                model_version_id=self.model_version_id or 1,
                input_features={
                    "irradiation": orig.irradiation,
                    "ambient_temperature": orig.ambient_temperature,
                    "module_temperature": orig.module_temperature,
                    "active_inverters": orig.active_inverters,
                },
            ))
            response_list.append(PredictionResponse(
                timestamp=orig.timestamp,
                plant_id=orig.plant_id,
                predicted_ac_power=pred_val,
                active_inverters=orig.active_inverters or 22,
                derating_applied=(orig.active_inverters is not None and orig.active_inverters < 22),
                model_version=version_str,
                status="SUCCESS",
            ))

        db.add_all(db_records)
        db.commit()

        return BatchPredictionResponse(
            predictions=response_list,
            total_records=len(response_list),
            model_version=version_str,
        )

    def analyze_deviation(self, request: DeviationAnalysisRequest, db: Session) -> DeviationAnalysisResponse:
        if not self._is_loaded or self.predictor is None or self.detector is None:
            raise RuntimeError("ML model or deviation detector not loaded.")

        pred_res = self.predictor.predict_single(
            timestamp=str(request.timestamp),
            plant_id=request.plant_id,
            irradiation=request.irradiation,
            ambient_temperature=request.ambient_temperature,
            module_temperature=request.module_temperature,
            active_inverters=request.active_inverters,
        )
        predicted_ac = pred_res["predicted_ac_power"]

        dev_result = self.detector.analyze_single(
            actual_ac_power=request.actual_ac_power,
            predicted_ac_power=predicted_ac,
            irradiation=request.irradiation,
            ambient_temperature=request.ambient_temperature,
            module_temperature=request.module_temperature,
            plant_id=request.plant_id,
            timestamp=str(request.timestamp),
            active_inverters=request.active_inverters,
        )

        version_str = self.metadata.get("model_version", "v1.1.0")
        if not self.model_version_id:
            self.sync_db_metadata(db)

        # Store prediction and deviation event in db
        pred_record = Prediction(
            timestamp=request.timestamp,
            plant_id=request.plant_id,
            actual_ac_power=request.actual_ac_power,
            predicted_ac_power=predicted_ac,
            model_version_id=self.model_version_id or 1,
            input_features={
                "irradiation": request.irradiation,
                "ambient_temperature": request.ambient_temperature,
                "module_temperature": request.module_temperature,
                "active_inverters": request.active_inverters,
            },
        )
        db.add(pred_record)
        db.commit()
        db.refresh(pred_record)

        dev_event = DeviationEvent(
            prediction_id=pred_record.id,
            residual=dev_result["residual"],
            absolute_deviation=dev_result["absolute_deviation"],
            deviation_percentage=dev_result["deviation_percentage"],
            severity=dev_result["severity"],
            contributing_conditions=dev_result["contributing_conditions"],
            investigation_status=dev_result["investigation_status"],
        )
        db.add(dev_event)
        db.commit()
        db.refresh(dev_event)

        return DeviationAnalysisResponse(
            id=dev_event.id,
            prediction_id=pred_record.id,
            timestamp=request.timestamp,
            plant_id=request.plant_id,
            actual_ac_power=request.actual_ac_power,
            predicted_ac_power=predicted_ac,
            residual=dev_result["residual"],
            absolute_deviation=dev_result["absolute_deviation"],
            deviation_percentage=dev_result["deviation_percentage"],
            severity=dev_result["severity"],
            contributing_conditions=dev_result["contributing_conditions"],
            investigation_status=dev_result["investigation_status"],
            active_inverters=request.active_inverters or 22,
            model_version=version_str,
        )

    def get_model_info(self) -> ModelInfoResponse:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded.")
        return ModelInfoResponse(
            model_name=self.metadata.get("model_name", "Solar_Power_Predictor"),
            model_version=self.metadata.get("model_version", "v1.1.0"),
            target_variable=self.metadata.get("target_variable", "AC_POWER"),
            training_timestamp=self.metadata.get("training_timestamp"),
            selected_features=self.metadata.get("features", []),
            feature_count=self.metadata.get("feature_count", 0),
            validation_metrics=self.metadata.get("test_metrics", {}),
            model_status="ACTIVE",
        )

    def get_statistics(self, db: Session) -> StatisticsResponse:
        total_p = db.query(func.count(Prediction.id)).scalar() or 0
        total_d = db.query(func.count(DeviationEvent.id)).scalar() or 0

        p1_count = db.query(func.count(Prediction.id)).filter(Prediction.plant_id == 4135001).scalar() or 0
        p2_count = db.query(func.count(Prediction.id)).filter(Prediction.plant_id == 4136001).scalar() or 0

        severities = db.query(DeviationEvent.severity, func.count(DeviationEvent.id)).group_by(DeviationEvent.severity).all()
        sev_dict = {s[0]: s[1] for s in severities}

        return StatisticsResponse(
            total_predictions=total_p,
            total_deviations=total_d,
            plant_prediction_counts={"Plant_1": p1_count, "Plant_2": p2_count},
            deviation_severity_counts=sev_dict,
        )


ml_service = MLInferenceService()
