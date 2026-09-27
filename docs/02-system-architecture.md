# 02 — System Architecture

## Overview

The customer-churn system is composed of five distinct layers that work together to take raw data all the way to a live prediction API. Each layer has a single responsibility and communicates with the others through well-defined interfaces (files, HTTP, or the MLflow tracking API).

```mermaid
graph TD
    subgraph "Data Layer"
        A[data_processor.py] -->|writes| B[data/raw/customers.csv]
        A -->|writes| C[data/processed/train.csv]
        A -->|writes| D[data/processed/test.csv]
    end

    subgraph "Training Layer"
        C -->|reads| E[train.py]
        D -->|reads| E
        E -->|logs params + metrics| F[(MLflow Tracking Store)]
        E -->|logs model artifact| F
    end

    subgraph "Serving Layer"
        F -->|loads latest model| G[api.py / FastAPI]
        H[HTTP Client] -->|POST /predict| G
        G -->|churn_probability + churn_label| H
    end

    subgraph "Versioning Layer"
        C -->|tracked by| I[DVC]
        D -->|tracked by| I
        I -->|pointer files| J[(Git Repository)]
        I -->|data contents| K[(DVC Remote)]
    end

    subgraph "CI/CD Layer"
        J -->|triggers| L[GitHub Actions]
        L -->|runs| M[pytest]
        L -->|builds + pushes| N[(GHCR Docker Image)]
    end
```

## Component Descriptions

### Data Layer — `data_processor.py`

The entry point for the entire pipeline. It generates a synthetic dataset, saves the raw CSV, and splits it into training and test sets. Nothing downstream can run without this step completing first.

**Inputs:** None (generates data programmatically)  
**Outputs:** `data/raw/customers.csv`, `data/processed/train.csv`, `data/processed/test.csv`

### Training Layer — `train.py`

Reads the processed training and test CSVs, trains a Random Forest classifier, evaluates it on the test set, and logs everything to MLflow. The trained model is stored as an MLflow artifact — not as a local file — making it accessible to any component that can reach the MLflow tracking store.

**Inputs:** `data/processed/train.csv`, `data/processed/test.csv`, `configs/config.yaml`  
**Outputs:** MLflow run (parameters, metrics, model artifact)

### Serving Layer — `api.py`

A FastAPI application that loads the latest completed MLflow run's model at startup and exposes a `/predict` endpoint. The model lives in `app.state.model` for the lifetime of the server process.

**Inputs:** MLflow tracking store (at startup), HTTP POST requests (at runtime)  
**Outputs:** JSON responses with `churn_probability` and `churn_label`

### Versioning Layer — DVC + `setup_data_tracking.sh`

DVC (Data Version Control) tracks the processed data directory. Instead of committing large CSV files to Git, DVC commits small pointer files (`.dvc` files) that record the data's hash and location. The actual data is stored in DVC's cache or a configured remote.

**Inputs:** `data/processed/` directory  
**Outputs:** `data/processed.dvc` pointer file, `.dvc/` metadata directory

### CI/CD Layer — `.github/workflows/ci.yml`

A GitHub Actions workflow with two jobs. The `test` job runs on every push and pull request to `main`. The `build-and-push` job runs only on pushes to `main` (after tests pass) and publishes the Docker image to GitHub Container Registry (GHCR).

**Inputs:** Git push or pull request event  
**Outputs:** Test results; Docker image at `ghcr.io/<owner>/customer-churn-api:latest`

---

## Request/Response Flow

This diagram traces a single prediction request from the HTTP client to the response.

```mermaid
sequenceDiagram
    participant Client as HTTP Client
    participant API as FastAPI (api.py)
    participant Model as RandomForestClassifier
    participant Pydantic as Pydantic Validator

    Client->>API: POST /predict {"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}
    API->>Pydantic: Validate CustomerFeatures
    Pydantic-->>API: Validated object (or 422 error)
    API->>Model: model.predict(features_dataframe)
    Model-->>API: [1]
    API->>Model: model.predict_proba(features_dataframe)
    Model-->>API: [[0.2, 0.8]]
    API-->>Client: {"churn_probability": 0.8, "churn_label": 1}
```

## API Startup Flow

Before the API can serve requests, it must load the model. This happens once when the server starts.

```mermaid
sequenceDiagram
    participant Uvicorn
    participant Lifespan as lifespan() context manager
    participant MLflow as MLflow Tracking Store

    Uvicorn->>Lifespan: Application startup
    Lifespan->>MLflow: set_tracking_uri()
    Lifespan->>MLflow: get_experiment_by_name("customer-churn")
    MLflow-->>Lifespan: Experiment object
    Lifespan->>MLflow: search_runs(status=FINISHED, order=DESC, limit=1)
    MLflow-->>Lifespan: DataFrame with run_id
    Lifespan->>MLflow: load_model("runs:/<run_id>/random_forest")
    MLflow-->>Lifespan: RandomForestClassifier
    Lifespan->>Uvicorn: Store model in app.state.model, yield
    Note over Uvicorn: Server is now ready to accept requests
```

## Data Flow Through the Pipeline

```mermaid
flowchart LR
    A["NumPy RNG\n(seed=42)"] --> B["1,000 rows\naccount_age, monthly_spend,\nsupport_tickets, churn"]
    B --> C["data/raw/customers.csv"]
    B --> D["stratified train_test_split\n80% / 20%"]
    D --> E["data/processed/train.csv\n800 rows"]
    D --> F["data/processed/test.csv\n200 rows"]
    E --> G["RandomForestClassifier.fit()"]
    F --> H["model.predict()"]
    G --> H
    H --> I["accuracy, precision, recall\nlogged to MLflow"]
    G --> J["model artifact\nlogged to MLflow"]
    J --> K["FastAPI /predict endpoint"]
```

---

Previous: [Project Overview ←](01-project-overview.md) | Next: [Project Structure →](03-project-structure.md)
