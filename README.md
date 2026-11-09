# Solar Power Generation Prediction and Performance Deviation Analysis-II

A solar generation prediction and performance-deviation analysis project using a trained machine-learning pipeline, a FastAPI backend, a SQL database, and a browser-based dashboard.

> **Baseline status:** This repository snapshot is the shared starting point for team development. The current automated test suite has 20 tests; the baseline fix in this commit addresses two history-endpoint tests that previously failed when called directly. Model artifacts were serialized with scikit-learn 1.9.0, so that version is pinned in the requirements files.

## Project structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/             # Health and versioned REST endpoints
│   │   ├── core/            # Application settings and paths
│   │   ├── database/        # SQLAlchemy engine and sessions
│   │   ├── models/          # Database entities
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/       # ML inference and persistence integration
│   │   └── static/          # Browser dashboard
│   ├── alembic/             # Database migration history
│   └── tests/               # Backend/API tests
├── ml/
│   ├── data/raw/            # Source solar generation and weather CSVs
│   ├── models/              # Trained model and preprocessing artifacts
│   ├── notebooks/           # Five exploratory and modelling notebooks
│   ├── outputs/             # Existing visualizations
│   ├── reports/             # Evaluation and analysis reports
│   ├── src/                 # Reusable ML pipeline code
│   └── tests/               # ML tests
├── requirements.txt
└── .env.example
```

## Requirements

- Python 3.12 or 3.13
- Git
- The Python packages in `requirements.txt`
- Optional: PostgreSQL, if you want to develop against PostgreSQL instead of the default local SQLite database

The trained model files were created using **scikit-learn 1.9.0**. Keep this version consistent when loading the committed `.joblib` artifacts.

## Setup on Windows PowerShell

Run these commands from the repository root (the folder containing this README):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. You can also run `.venv\Scripts\python.exe` directly without activating the environment.

## Run the application

From the repository root, with the virtual environment active:

```powershell
uvicorn app.main:app --app-dir backend --reload
```

Open these URLs:

- Dashboard: http://127.0.0.1:8000/
- API documentation (Swagger): http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

On startup, the backend creates missing database tables in the configured database, loads the committed model artifacts, and syncs basic plant/model metadata. With no `DATABASE_URL` override, it uses a local SQLite file named `solar_dev.db` in the repository root. Local database files are ignored by Git.

## Run tests

From the repository root:

```powershell
python -m pytest backend/tests ml/tests -v
```

The tests use a local SQLite test database. Do not commit generated `*.db` files.

## Work with the ML notebooks

Install the project requirements, then launch Jupyter from the repository root:

```powershell
pip install jupyter
jupyter lab
```

Open the notebooks in `ml/notebooks/` in numerical order. Keep the working directory at the repository root because the ML scripts and notebooks use project-relative paths such as `ml/data/raw`.

To run the ML test suite only:

```powershell
python -m pytest ml/tests -v
```

## API overview

The versioned API is mounted under `/api/v1`. The current routes include:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v1/model/info` | Active model details and metrics |
| POST | `/api/v1/predict` | Predict one solar generation record |
| POST | `/api/v1/predict/batch` | Predict a batch of records |
| POST | `/api/v1/deviations/analyze` | Compare actual and predicted generation |
| GET | `/api/v1/predictions` | Query prediction history |
| GET | `/api/v1/deviations` | Query deviation history |
| GET | `/api/v1/statistics` | Database and operational summary |

Use `/docs` for the exact request and response schemas. Do not assume an endpoint's request fields: inspect the Pydantic schemas in `backend/app/schemas/solar.py` or the generated API documentation.

## Git collaboration workflow

- `main` is the stable branch.
- `saketh-integration` is the shared integration branch.
- Each contributor works on their own role branch, for example `role-2-ml`, `role-3-backend`, or `role-4-frontend`.
- Open pull requests into `saketh-integration`, not directly into `main`.
- Run relevant tests before requesting review. Coordinate changes to shared files such as requirements, schemas, configuration, and this README.

### Suggested ownership boundaries

| Role | Main ownership |
|---|---|
| Role 1 — Project lead/integration | Architecture, API contracts, ML/backend integration, cross-team review, Git and deployment coordination |
| Role 2 — ML | `ml/` |
| Role 3 — Backend/database | `backend/app/api/`, `backend/app/models/`, `backend/app/schemas/`, `backend/app/services/`, `backend/tests/` |
| Role 4 — Frontend/dashboard | `backend/app/static/` |

If a task requires changing a file outside your ownership area, discuss it first and coordinate the interface with the owner.

## Important project notes

- The supplied dataset covers two solar plants and includes generation and weather sensor data.
- A deviation is an indication for investigation, not proof of an equipment fault.
- The current repository snapshot does not include a maintained Docker Compose deployment setup; deployment configuration should be added and tested as a team task rather than assumed to be ready.
- Do not commit secrets, local database files, Python environments, cache folders, or personal datasets.
