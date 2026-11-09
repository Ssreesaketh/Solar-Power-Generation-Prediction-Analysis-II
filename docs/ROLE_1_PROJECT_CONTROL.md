# Role 1 — Project Coordination and Integration Control

**Project:** Solar Power Generation Prediction and Performance Deviation Analysis-II  
**Owner:** Role 1 — Project Lead & Integration  
**Baseline reviewed:** `main` at `bdd6774651d0257bc2310350f5e1c0ce1ebf518b`  
**Working branch:** `role-1-sreesaketh`  
**Integration branch:** `saketh-integration`

## 1. Purpose

This document records the Role 1 project baseline, ownership boundaries, Git workflow, integration contracts, risks and the project-level completion checklist. It is a coordination document; component implementation remains with the assigned role unless a cross-team change is agreed.

## 2. Git branch policy

| Branch | Purpose | Rule |
|---|---|---|
| `main` | Stable baseline and final releases | No direct feature commits |
| `saketh-integration` | Shared integration target | Merge reviewed, tested role PRs here |
| `role-1-sreesaketh` | Role 1 planning, coordination and integration work | PR target: `saketh-integration` |
| `role-2-ml` | ML implementation | To be created/used by Role 2 |
| `role-3-backend` | API and database implementation | To be created/used by Role 3 |
| `role-4-frontend` | Dashboard implementation | To be created/used by Role 4 |

Workflow:

1. Pull the latest `saketh-integration` before starting a task.
2. Work only on the contributor's role branch.
3. Keep commits focused and use descriptive messages.
4. Run the relevant tests before opening a pull request.
5. Open pull requests into `saketh-integration`, never directly into `main`.
6. Role 1 checks the changed files, contract compatibility, tests and conflicts before merging.
7. Promote a tested integration snapshot to `main` only after end-to-end verification.

## 3. Component ownership

| Role | Ownership | Main deliverables |
|---|---|---|
| Role 1 — Project lead/integration | Project-level coordination; cross-component contracts; Git and release flow | Baseline audit, task tracking, integration reviews, deployment coordination, final README |
| Role 2 — ML | `ml/` | Five notebooks, reproducible training/evaluation, model artifacts and deviation analysis |
| Role 3 — Backend | `backend/app/api/`, `backend/app/models/`, `backend/app/schemas/`, `backend/app/services/`, `backend/tests/`, migrations | Stable API, inference integration, persistence, migrations and backend tests |
| Role 4 — Frontend | `backend/app/static/` | Dashboard, API-connected charts, filters, loading/error/empty states and responsive layout |

Shared files such as root `README.md`, `requirements.txt`, `.env.example`, schemas and deployment configuration must be coordinated before editing.

## 4. Verified repository baseline

The current repository contains:

- `backend/`: FastAPI application, Pydantic schemas, SQLAlchemy models/session, ML inference service, Alembic configuration/migration, one backend test module and the HTML dashboard.
- `ml/`: raw generation/weather CSVs for two plants, five numbered notebooks, reusable Python modules, model artifacts, evaluation reports, visualizations and ML tests.
- Root setup files: `README.md`, `requirements.txt`, `.env.example` and `.gitignore`.
- The five notebooks are `01_eda.ipynb`, `02_preprocessing.ipynb`, `03_feature_engineering.ipynb`, `04_model_comparison.ipynb` and `05_deviation_analysis.ipynb`.

The root README reports a baseline suite of 20 tests. This is recorded as repository documentation, not as a fresh test run by Role 1.

## 5. Initial integration contracts

### ML to backend

The backend request schema currently accepts:

- `timestamp` (datetime)
- `plant_id` (4135001 or 4136001)
- `irradiation`
- `ambient_temperature`
- `module_temperature`
- optional `active_inverters` (default 22; range 0–22)

Batch requests contain 1–500 records. Deviation analysis additionally accepts `actual_ac_power`.

The model metadata declares 15 engineered features and identifies `AC_POWER` in kW as the target. The backend service calls the ML predictor and persists prediction/deviation records. Role 2 must provide Role 3 with the final authoritative input/output example and artifact/version details before integration is considered complete.

### Backend to dashboard

The root README documents these API routes:

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/v1/model/info` | Active model information and metrics |
| POST | `/api/v1/predict` | Single prediction |
| POST | `/api/v1/predict/batch` | Batch prediction |
| POST | `/api/v1/deviations/analyze` | Actual-versus-predicted deviation analysis |
| GET | `/api/v1/predictions` | Prediction history |
| GET | `/api/v1/deviations` | Deviation history |
| GET | `/api/v1/statistics` | Operational summary |

Role 3 owns the authoritative API schemas. Role 4 should use the actual OpenAPI/schema responses rather than assuming field names or shapes.

## 6. Initial cross-component findings and risks

| ID | Finding from current files | Required action | Owner | Status |
|---|---|---|---|---|
| R1-01 | Current `model_metadata.json` and `ml/README.md` both identify a Voting Ensemble. The older task brief mentions a Gradient Boosting documentation conflict, which is not present in these two current files. | Role 2 verifies the serialized artifact and confirms one authoritative algorithm name in all notebooks, reports and metadata. | Role 2 | Verify artifact |
| R1-02 | ML README headline metrics differ from current metadata/report metrics. Metadata reports overall MAE 486.096 and R² 0.964; evaluation JSON reports MAE 486.0164 and R² 0.964. ML README still reports MAE 587.21 and R² 0.9602. | Reproduce evaluation and update documentation from the authoritative run. | Role 2 | Open |
| R1-03 | High-power subgroup has only 42 samples and reported R² is -0.2606; Plant 2 R² is 0.9000 in metadata. | Investigate subgroup error and explain limitations; do not present overall score alone. | Role 2 | Open |
| R1-04 | Alembic configuration and an initial migration exist, while application startup also calls `Base.metadata.create_all`. | Role 3 checks migration state and defines one repeatable database initialization/migration workflow. | Role 3 | Open |
| R1-05 | Model metadata declares 15 engineered features, while the API accepts raw environmental telemetry plus active inverter count. | Role 2 and Role 3 verify preprocessing parity and document which fields are used for model inference versus post-prediction derating. | Roles 2 + 3 | Open |
| R1-06 | Dashboard is currently served from `backend/app/static/index.html`; the root README describes a browser-based dashboard. | Role 4 inventories current live API calls and placeholder/sample sections before changing UI. | Role 4 | Open |
| R1-07 | README says deployment configuration is not maintained and should be added/tested. | Role 1 + Role 3 define the simplest reproducible deployment setup after local integration works. | Roles 1 + 3 | Open |

Important: these are audit findings and verification tasks, not proof that the affected components are broken.

## 7. Role 1 task checklist

### Phase 1 — Audit and stabilization

- [x] Confirm repository, default branch and baseline commit.
- [x] Create shared integration branch.
- [x] Create Role 1 working branch.
- [x] Record component ownership and PR policy.
- [x] Inspect repository structure and identify cross-component contracts.
- [ ] Collect Role 2, Role 3 and Role 4 audit findings.
- [ ] Confirm each contributor has a separate working branch.
- [ ] Agree on one API request/response example for prediction and deviation analysis.
- [ ] Finalize prioritized issue list with owner, evidence and acceptance criteria.

### Phase 2 — Component completion

- [ ] Review each role's pull requests and test evidence.
- [ ] Check ML/backend/frontend contracts before merging.
- [ ] Track shared-file changes and resolve integration conflicts.
- [ ] Keep `saketh-integration` runnable after each accepted merge.
- [ ] Require a working component demonstration before end-to-end integration.

### Phase 3 — End-to-end integration and release

- [ ] Verify plant selection → input → prediction → persistence → deviation analysis → dashboard.
- [ ] Verify model artifact loading and database startup from a clean environment.
- [ ] Coordinate deployment configuration and clean setup instructions.
- [ ] Run the complete test suite and record actual results.
- [ ] Review final documentation and screenshots.
- [ ] Rehearse the demonstration.
- [ ] Promote the verified release from `saketh-integration` to `main`.

## 8. Definition of done for Role 1 integration

A change is ready to merge only when:

1. Its owner and scope are clear.
2. It does not silently change an agreed API or model contract.
3. Relevant tests pass, with test output or CI evidence linked in the PR.
4. Any required frontend/backend/ML changes are coordinated.
5. The integration branch remains runnable, or any blocker is documented with an owner.
6. The PR description states what changed, how it was tested and any known limitations.
