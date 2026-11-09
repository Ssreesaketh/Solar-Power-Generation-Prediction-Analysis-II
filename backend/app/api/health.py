"""
System Health and Diagnostics Endpoint.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.solar import HealthResponse
from app.services.ml_service import ml_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="API and Infrastructure Health Check")
def check_health(db: Session = Depends(get_db)):
    """
    Validates backend connectivity, database status, and ML model loaded status.
    """
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        db_connected = False

    model_loaded = ml_service.is_loaded
    active_version = ml_service.metadata.get("model_version", "unknown")

    status_str = "HEALTHY" if (db_connected and model_loaded) else "DEGRADED"

    return HealthResponse(
        status=status_str,
        database_connected=db_connected,
        model_loaded=model_loaded,
        active_model_version=active_version,
        timestamp=datetime.now(timezone.utc),
    )
