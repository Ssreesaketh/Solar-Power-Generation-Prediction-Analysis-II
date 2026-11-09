"""
Comprehensive Backend API and Service Integration Tests.
Validates endpoints, schemas, database persistence, and ML service inference.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.database.session import Base
from app.services.ml_service import ml_service
from app.api.health import check_health
from app.api.v1.endpoints import (
    get_model_info,
    predict_single,
    predict_batch,
    analyze_deviation,
    list_predictions,
    list_deviations,
    get_statistics,
)
from app.schemas.solar import (
    PredictionRequest,
    BatchPredictionRequest,
    DeviationAnalysisRequest,
)
from app.main import serve_index

# Isolated SQLite test database
TEST_DB_URL = "sqlite:///./test_solar.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

NOON_TIMESTAMP = datetime(2020, 5, 15, 12, 0, 0, tzinfo=timezone.utc)
NIGHT_TIMESTAMP = datetime(2020, 5, 15, 1, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    ml_service.load_artifacts(db)
    db.close()
    yield
    test_engine.dispose()
    if os.path.exists("./test_solar.db"):
        try:
            os.remove("./test_solar.db")
        except Exception:
            pass


@pytest.fixture
def db():
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_health_endpoint(db):
    res = check_health(db=db)
    assert res.status == "HEALTHY"
    assert res.database_connected is True
    assert res.model_loaded is True
    assert res.active_model_version == "v1.1.0"


def test_model_info_endpoint():
    info = get_model_info()
    assert info.model_status == "ACTIVE"
    assert info.target_variable == "AC_POWER"
    assert len(info.selected_features) > 0
    assert "overall" in info.validation_metrics


def test_predict_single_daylight(db):
    req = PredictionRequest(
        timestamp=NOON_TIMESTAMP,
        plant_id=4135001,
        irradiation=0.85,
        ambient_temperature=34.0,
        module_temperature=52.0,
        active_inverters=22,
    )
    res = predict_single(request=req, db=db)
    assert res.predicted_ac_power > 0.0
    assert res.status == "SUCCESS"
    assert res.plant_id == 4135001
    assert res.active_inverters == 22
    assert res.derating_applied is False


def test_predict_single_capacity_derating(db):
    req_full = PredictionRequest(
        timestamp=NOON_TIMESTAMP,
        plant_id=4135001,
        irradiation=0.85,
        ambient_temperature=34.0,
        module_temperature=52.0,
        active_inverters=22,
    )
    res_full = predict_single(request=req_full, db=db)

    req_derated = PredictionRequest(
        timestamp=NOON_TIMESTAMP,
        plant_id=4135001,
        irradiation=0.85,
        ambient_temperature=34.0,
        module_temperature=52.0,
        active_inverters=11,  # 50% capacity
    )
    res_derated = predict_single(request=req_derated, db=db)
    assert round(res_derated.predicted_ac_power / res_full.predicted_ac_power, 1) == 0.5
    assert res_derated.derating_applied is True
    assert "50.0%" in res_derated.derating_details


def test_predict_single_nighttime_gating(db):
    req = PredictionRequest(
        timestamp=NIGHT_TIMESTAMP,
        plant_id=4135001,
        irradiation=0.0,
        ambient_temperature=23.0,
        module_temperature=22.0,
    )
    res = predict_single(request=req, db=db)
    assert res.predicted_ac_power == 0.0


def test_predict_batch(db):
    records = [
        PredictionRequest(
            timestamp=NOON_TIMESTAMP,
            plant_id=4135001,
            irradiation=0.5,
            ambient_temperature=30.0,
            module_temperature=40.0,
        ),
        PredictionRequest(
            timestamp=NOON_TIMESTAMP,
            plant_id=4136001,
            irradiation=0.9,
            ambient_temperature=36.0,
            module_temperature=55.0,
        ),
    ]
    batch_req = BatchPredictionRequest(records=records)
    res = predict_batch(batch_request=batch_req, db=db)
    assert res.total_records == 2
    assert len(res.predictions) == 2
    assert res.predictions[0].predicted_ac_power > 0.0


def test_predict_invalid_plant_id():
    with pytest.raises(ValidationError):
        PredictionRequest(
            timestamp=NOON_TIMESTAMP,
            plant_id=9999999,  # Unsupported plant
            irradiation=0.5,
            ambient_temperature=30.0,
            module_temperature=40.0,
        )


def test_predict_out_of_range_irradiation():
    with pytest.raises(ValidationError):
        PredictionRequest(
            timestamp=NOON_TIMESTAMP,
            plant_id=4135001,
            irradiation=-0.5,  # Invalid negative
            ambient_temperature=30.0,
            module_temperature=40.0,
        )


def test_deviation_analysis_normal(db):
    req = DeviationAnalysisRequest(
        timestamp=NOON_TIMESTAMP,
        plant_id=4135001,
        actual_ac_power=24000.0,
        irradiation=0.85,
        ambient_temperature=34.0,
        module_temperature=52.0,
    )
    res = analyze_deviation(request=req, db=db)
    assert "residual" in res.model_dump()
    assert res.severity in ["Normal", "Moderate Deviation"]


def test_deviation_analysis_significant_deficit(db):
    req = DeviationAnalysisRequest(
        timestamp=NOON_TIMESTAMP,
        plant_id=4135001,
        actual_ac_power=50.0,  # Extreme underperformance under bright sun
        irradiation=0.9,
        ambient_temperature=35.0,
        module_temperature=58.0,
    )
    res = analyze_deviation(request=req, db=db)
    assert res.severity in ["Significant Deviation", "Extreme Deviation"]
    assert len(res.contributing_conditions) > 0


def test_predictions_history(db):
    data = list_predictions(page=1, page_size=10, db=db)
    assert data["total"] >= 1
    assert len(data["records"]) >= 1


def test_deviations_history_and_filters(db):
    data = list_deviations(page=1, page_size=10, db=db)
    assert data["total"] >= 1
    assert len(data["records"]) >= 1

    filtered = list_deviations(plant_id=4135001, db=db)
    assert filtered["total"] >= 1


def test_database_statistics(db):
    stats = get_statistics(db=db)
    assert stats.total_predictions >= 1
    assert stats.total_deviations >= 1
    assert "Plant_1" in stats.plant_prediction_counts


def test_serve_html_index():
    response = asyncio.run(serve_index())
    assert response.status_code == 200
