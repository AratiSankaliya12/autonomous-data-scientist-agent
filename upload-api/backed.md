# Autonomous Data Scientist — Backend

This document describes how the **upload-api** backend is structured and how the **agentic data scientist** pipeline works end to end.

## Purpose

The API lets you:

1. **Upload** tabular datasets (CSV, Excel, JSON, Parquet).
2. **Run a multi-step analysis** where specialized modules (“agents”) produce structured results in sequence—similar to how an autonomous data scientist would explore data, suggest cleaning, profile features, fit a baseline model, evaluate it, and prepare chart data for the UI.

This is **agentic** in the workflow sense: fixed roles, clear hand-offs, and machine-readable outputs per step. It does **not** require an external LLM; each agent is deterministic code (pandas, NumPy, scikit-learn).

## Directory layout

```
upload-api/
  main.py              # Uvicorn entry: `uvicorn main:app --reload`
  requirements.txt
  backed.md            # This file
  uploads/             # Stored files + index.json (created at runtime)
  app/
    main.py            # FastAPI app, CORS, router registration
    config.py          # Paths, size limits, CORS_ORIGINS
    schemas/           # Pydantic models (uploads, analyze request/response)
    routers/           # HTTP routes (health, uploads, analysis)
    services/          # Index + filesystem, dataframe load/preview helpers
    agents/            # One module per “agent” + orchestrator
```

Legacy folders at the repo root (`agents/`, `pipline/`) are not used by this service; the live implementation lives under `app/agents/`.

## HTTP API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/api/uploads` | List uploads with optional preview metadata |
| POST | `/api/upload` | Multipart file upload |
| DELETE | `/api/uploads/{id}` | Remove upload and index entry |
| POST | `/api/uploads/{id}/analyze` | Run the agent pipeline (see below) |

### Analyze request body (`POST /api/uploads/{id}/analyze`)

```json
{
  "target_column": "optional_column_name",
  "steps": null,
  "apply_cleaning": false
}
```

- **`target_column`**: If set, the **modeling** and **evaluation** agents run a baseline Random Forest (classification or regression, inferred from the target). If omitted, those steps are **skipped** (modeling) or **skipped** (evaluation).
- **`steps`**: `null` runs all agents that apply. Otherwise pass a subset, e.g. `["exploration", "visualization"]`.
- **`apply_cleaning`**: If `true`, the cleaning agent **mutates a copy** of the data in memory for subsequent steps (dedupe + simple imputation). The original file on disk is unchanged.

### Analyze response

Returns `upload_id`, timestamps, `target_column`, and a **`steps`** array. Each step has:

- `agent`: `exploration` | `cleaning` | `features` | `modeling` | `evaluation` | `visualization`
- `status`: `ok` | `skipped` | `error`
- `summary`: Short human-readable line
- `payload`: Structured JSON (metrics, chart data, etc.)
- `error`: Present when `status` is `error`

## The agent pipeline (how it “thinks”)

The **orchestrator** (`app/agents/orchestrator.py`) loads the uploaded file into a pandas `DataFrame`, then runs agents **in order**:

1. **Exploration** — Row/column counts, duplicate rows, numeric `describe`, small value-count samples for low-cardinality columns.
2. **Cleaning** — Detects duplicates and missing values; optionally applies in-memory dedupe + median/mode imputation.
3. **Features** — Lists numeric vs non-numeric columns, flags high-cardinality categoricals and constant columns.
4. **Modeling** — If `target_column` is set: builds a preprocessing pipeline (median imputation for numeric, most-frequent + one-hot for categoricals) and fits a **RandomForest** baseline; reports hold-out metrics and top feature importances when available.
5. **Evaluation** — Summarizes the same hold-out metrics in a dedicated step (for clearer UI and future extension, e.g. calibration or fairness).
6. **Visualization** — Builds **chart-ready data** in `payload.charts` (e.g. missing-value bar series, histogram bin/counts for the first numeric column). The frontend can render these without matplotlib on the server.

Failures in one agent do not crash the whole app: the step is marked `error` with a message, and later steps still run where it makes sense.

## Configuration

- **`CORS_ORIGINS`**: Comma-separated list in the environment variable (default: `http://localhost:3000,http://127.0.0.1:3000`).
- **Upload limits**: `MAX_BYTES` and `ALLOWED_EXT` in `app/config.py`.

## Run locally

```bash
cd upload-api
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive OpenAPI.

## Extending toward LLM-based agents

To plug in true LLM agents later, keep the same **orchestrator contract**: each agent accepts context (dataframe metadata, prior payloads) and returns `summary` + `payload`. You can add a step that calls an API (OpenAI, etc.) to turn structured metrics into narratives or to propose next steps, without changing the upload or analyze routes.
