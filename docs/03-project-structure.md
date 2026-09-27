# 03 — Project Structure

## Repository Layout

```
ml-cookbook/                          ← Repository root
├── .github/
│   └── workflows/
│       └── ci.yml                    ← GitHub Actions CI/CD pipeline
├── customer-churn/                   ← The first MLOps blueprint
│   ├── configs/
│   │   └── config.yaml               ← Centralised hyperparameters and file paths
│   ├── src/
│   │   ├── api.py                    ← FastAPI prediction service
│   │   ├── data_processor.py         ← Data generation and splitting
│   │   ├── evaluate.py               ← Standalone model evaluation script
│   │   ├── predict_remote.py         ← Fetch model from MLflow and predict
│   │   └── train.py                  ← Model training and MLflow logging
│   ├── tests/
│   │   ├── test_api.py               ← API endpoint tests
│   │   └── test_data.py              ← Data generation tests
│   ├── .dockerignore                 ← Files excluded from the Docker build context
│   ├── .env.example                  ← Template for environment variables
│   ├── Dockerfile                    ← Container build instructions
│   ├── README.md                     ← Project-level setup and usage guide
│   ├── requirements.txt              ← Python dependencies
│   └── setup_data_tracking.sh        ← Initialises DVC for data versioning
├── .gitignore                        ← Files excluded from Git tracking
└── README.md                         ← Repository-level overview
```

## Generated Directories (Not Committed to Git)

These directories are created at runtime and are excluded by `.gitignore`:

```
customer-churn/
├── data/
│   ├── raw/
│   │   └── customers.csv             ← Full 1,000-row synthetic dataset
│   └── processed/
│       ├── train.csv                 ← 800-row training split
│       └── test.csv                  ← 200-row test split
├── models/                           ← Used by evaluate.py (local joblib model)
├── mlruns/                           ← MLflow local tracking store (if not using Databricks)
└── mlflow.db                         ← SQLite MLflow backend (if not using Databricks)
```

## File-by-File Reference

### `configs/config.yaml`

The single source of truth for all tunable settings. Both `data_processor.py` and `train.py` read this file at runtime. Changing a value here affects the entire pipeline without touching source code.

See [Configuration →](05-configuration.md) for a full breakdown.

### `src/data_processor.py`

Generates the synthetic dataset and writes three CSV files. It is the first script to run in the pipeline and has no dependencies on other `src/` modules.

See [Data Pipeline →](06-data-pipeline.md) for a full walkthrough.

### `src/train.py`

Reads the processed CSVs, trains the Random Forest, and logs everything to MLflow. It depends on `data_processor.py` having run first.

See [Model Training →](07-model-training.md) for a full walkthrough.

### `src/evaluate.py`

A standalone script that loads a model from a local `.joblib` file and prints a classification report. This is independent of MLflow and is useful for quick local evaluation.

See [Model Evaluation →](08-model-evaluation.md) for details.

### `src/api.py`

The FastAPI application. It loads the latest MLflow model at startup and serves predictions at `POST /predict`. This is the main deliverable of the project.

See [Prediction API →](09-prediction-api.md) for a full walkthrough.

### `src/predict_remote.py`

A script that connects to the configured MLflow tracking store, downloads the latest completed model, and runs a single prediction. Useful for verifying that a remote MLflow deployment is working.

See [Remote Prediction →](10-remote-prediction.md) for details.

### `tests/test_api.py`

Tests the `/predict` endpoint using FastAPI's `TestClient`. Uses `monkeypatch` to replace all MLflow calls with stubs, so no real MLflow server is needed.

### `tests/test_data.py`

Tests the `generate_data` function to verify the output shape, column names, and absence of null values.

### `Dockerfile`

Builds a self-contained Docker image. The build process installs dependencies, copies source code, generates data, and trains a model — so the image ships with a ready-to-serve model baked in.

See [Docker and Containerisation →](13-docker.md) for details.

### `setup_data_tracking.sh`

A Bash script that initialises DVC and adds the `data/processed/` directory to DVC tracking. It prints the Git commands needed to commit the resulting metadata files.

See [Data Versioning with DVC →](11-data-versioning.md) for details.

### `.env.example`

A template showing the four environment variables the project needs. Copy this to `.env` and fill in real values. The `.env` file is excluded from Git by `.gitignore`.

### `requirements.txt`

Lists all Python dependencies without pinned versions. This keeps the project easy to install while relying on the Docker image for reproducible production deployments.

### `.gitignore`

Excludes generated data, model files, MLflow artefacts, Python cache directories, and the `.env` file. Notably, it uses a pattern that allows the DVC pointer file (`data/processed.dvc`) to be committed while blocking the actual data files.

### `.dockerignore`

Excludes the same generated files from the Docker build context, preventing large data and model files from being sent to the Docker daemon unnecessarily.

### `.github/workflows/ci.yml`

Defines two GitHub Actions jobs: `test` (runs on all pushes and PRs to `main`) and `build-and-push` (runs only on pushes to `main` after tests pass).

See [CI/CD with GitHub Actions →](14-cicd.md) for details.

---

Previous: [System Architecture ←](02-system-architecture.md) | Next: [Technology Stack →](04-technology-stack.md)
