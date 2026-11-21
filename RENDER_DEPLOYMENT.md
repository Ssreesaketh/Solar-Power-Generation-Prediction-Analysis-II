# Render deployment

This repository is deployed as one Render Web Service. The existing FastAPI application serves the web UI from `backend/app/static/index.html`.

## Render configuration
The root-level `render.yaml` contains the service configuration. Render uses the repository root, installs `requirements.txt`, and starts `app.main:app` from the `backend` directory.

## 15-minute two-plant forecast model
The additional forecast endpoints are:
- `GET /forecast-health`
- `GET /sample-request?plant_id=4135001` (synthetic sample history for interface testing only)
- `POST /predict` (requires exactly 97 consecutive 15-minute readings)

The trained two-plant artifact must be placed at `ml/models/plant_forecast_rf.pkl`, or the Render environment variable `FORECAST_MODEL_PATH` must point to it. The artifact is intentionally not substituted with the existing Phase-I model: they have different feature contracts.

**Important:** the sample-request endpoint creates synthetic history and is not suitable for operational forecasts. Meaningful predictions require actual consecutive plant telemetry.
