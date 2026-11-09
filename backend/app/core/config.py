"""
Application Configuration Module.
Loads environment variables using Pydantic Settings.
"""

import os
import sys
import tempfile
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Prevent loky subprocess wmic hang on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

# Locate project root (containing ml/)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent.parent
if not (PROJECT_ROOT / "ml").exists():
    PROJECT_ROOT = Path(".").resolve()

# Ensure ml/src is in sys.path so joblib can deserialize custom transformers
ml_src_path = str((PROJECT_ROOT / "ml" / "src").resolve())
if ml_src_path not in sys.path:
    sys.path.insert(0, ml_src_path)


def resolve_model_path(relative_path: str) -> str:
    candidates = [
        Path(relative_path),
        PROJECT_ROOT / relative_path,
        PROJECT_ROOT.parent / relative_path,
        Path("..") / relative_path,
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str((PROJECT_ROOT / relative_path).resolve())


class Settings(BaseSettings):
    PROJECT_NAME: str = "Solar Power Generation & Deviation API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Database Configuration
    DATABASE_URL: str = "sqlite:///./solar_dev.db"
    DB_POOL_PRE_PING: bool = True

    # ML Model Paths (Resolved relative to project root)
    MODEL_PATH: str = resolve_model_path("ml/models/best_model.joblib")
    PIPELINE_PATH: str = resolve_model_path("ml/models/preprocessing_pipeline.joblib")
    DETECTOR_PATH: str = resolve_model_path("ml/models/deviation_detector.joblib")
    METADATA_PATH: str = resolve_model_path("ml/models/model_metadata.json")

    # Batch Request Constraints
    MAX_BATCH_SIZE: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
