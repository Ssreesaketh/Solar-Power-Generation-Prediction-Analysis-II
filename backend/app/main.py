"""
FastAPI Backend Application Entrypoint.
Initializes lifespan, database tables, ML inference service, API routers,
and serves the interactive testing web interface.
"""

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.database.session import engine, Base, SessionLocal
from app.api.health import router as health_router
from app.api.v1.endpoints import router as v1_router
from app.services.ml_service import ml_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("solar_api")

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)

    logger.info("Loading ML Model and Preprocessor artifacts...")
    db = SessionLocal()
    try:
        ml_service.load_artifacts(db)
    finally:
        db.close()

    yield
    # Shutdown sequence
    logger.info("Shutting down Solar Power Prediction API.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production REST API for Solar Power Generation Estimation, "
        "Diurnal Performance Tracking, and Anomaly Deviation Diagnosis."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS middleware for local frontend & remote clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please verify inputs or contact administrator."},
    )


# Serve interactive HTML testing interface at root
@app.get("/", summary="Serve Interactive Solar Testing Interface", include_in_schema=False)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Solar Power API is running. UI not found."}


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount API routers
app.include_router(health_router, tags=["Health & Diagnostics"])
app.include_router(v1_router, prefix=settings.API_V1_STR, tags=["Solar Power & Deviation Telemetry"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
