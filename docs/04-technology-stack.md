# 04 — Technology Stack

Every library and tool in this project was chosen for a specific reason. This document explains what each one is, why it was chosen, and where it appears in the codebase.

## Python Libraries

### pandas

**What it is:** A data manipulation library that provides the `DataFrame` — a table-like data structure with labelled rows and columns.

**Why it is used:** Reading and writing CSV files, slicing feature columns from the dataset, and constructing the single-row DataFrame that the API sends to the model for prediction.

**Where it appears:**
- `data_processor.py` — builds the dataset as a DataFrame and writes it to CSV
- `train.py` — reads training and test CSVs into DataFrames
- `evaluate.py` — reads the test CSV
- `api.py` — constructs a one-row DataFrame from the incoming request
- `predict_remote.py` — constructs the sample customer DataFrame

### NumPy

**What it is:** A numerical computing library providing multi-dimensional arrays and mathematical functions.

**Why it is used:** Generating the synthetic dataset. NumPy's random number generator (`np.random.default_rng`) produces the feature values, and its mathematical functions (`np.exp`, `np.clip`, `np.round`) implement the logistic churn model.

**Where it appears:**
- `data_processor.py` — all data generation logic

### scikit-learn

**What it is:** The standard Python machine learning library. Provides algorithms, preprocessing tools, model evaluation utilities, and a consistent API.

**Why it is used:** Three specific components are used:
- `RandomForestClassifier` — the model
- `train_test_split` — splitting the dataset into training and test sets
- `accuracy_score`, `precision_score`, `recall_score`, `classification_report` — evaluation metrics

**Where it appears:**
- `data_processor.py` — `train_test_split`
- `train.py` — `RandomForestClassifier`, metric functions
- `evaluate.py` — `classification_report`

### PyYAML

**What it is:** A library for reading and writing YAML files.

**Why it is used:** Loading `configs/config.yaml` into a Python dictionary.

**Where it appears:**
- `data_processor.py`, `train.py`, `evaluate.py` — all call `yaml.safe_load()`

### FastAPI

**What it is:** A modern Python web framework for building HTTP APIs. It uses Python type hints to automatically validate request and response data, and generates interactive API documentation.

**Why it is used:** Serving the trained model as a REST API. FastAPI's Pydantic integration means that invalid requests (e.g., a negative `account_age`) are automatically rejected with a descriptive error before the prediction code even runs.

**Where it appears:**
- `api.py` — the entire serving layer

### Uvicorn

**What it is:** An ASGI (Asynchronous Server Gateway Interface) web server. FastAPI applications need an ASGI server to handle HTTP connections.

**Why it is used:** Running the FastAPI application in both development and production (inside Docker).

**Where it appears:**
- `Dockerfile` — the `CMD` instruction
- `customer-churn/README.md` — the `uvicorn` command for local development

### Pydantic

**What it is:** A data validation library that uses Python type hints to define data schemas. FastAPI uses Pydantic internally.

**Why it is used:** Defining the request schema (`CustomerFeatures`) and response schema (`ChurnPrediction`) for the API. Pydantic enforces constraints like `gt=0` (greater than zero) and `ge=0` (greater than or equal to zero) automatically.

**Where it appears:**
- `api.py` — `CustomerFeatures` and `ChurnPrediction` model classes

### MLflow

**What it is:** An open-source platform for managing the machine learning lifecycle. It provides experiment tracking (logging parameters, metrics, and artifacts), a model registry, and model serving utilities.

**Why it is used:** Tracking every training run so that results are reproducible and comparable. The trained model is stored as an MLflow artifact, which allows `api.py` and `predict_remote.py` to load it by run ID without needing to know where the file is stored.

**Where it appears:**
- `train.py` — `mlflow.start_run()`, `mlflow.log_params()`, `mlflow.log_metrics()`, `mlflow.sklearn.log_model()`
- `api.py` — `mlflow.get_experiment_by_name()`, `mlflow.search_runs()`, `mlflow.sklearn.load_model()`
- `predict_remote.py` — `mlflow.pyfunc.load_model()`

### Databricks SDK (`databricks-sdk`)

**What it is:** The official Python SDK for the Databricks platform. When `MLFLOW_TRACKING_URI=databricks`, MLflow uses this SDK to authenticate with the Databricks workspace.

**Why it is used:** Databricks Community Edition provides a free hosted MLflow tracking server. The SDK handles authentication using `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment variables.

**Where it appears:**
- `requirements.txt` — listed as a dependency
- Implicitly used by MLflow when `MLFLOW_TRACKING_URI=databricks`

### DVC (Data Version Control)

**What it is:** A version control system for data and machine learning models. It works alongside Git: DVC stores small pointer files in Git while keeping large data files in a separate cache or remote storage.

**Why it is used:** Tracking the processed dataset so that the exact data used for any training run can be reproduced. Without DVC, the CSV files would either be committed to Git (bloating the repository) or not tracked at all (making runs irreproducible).

**Where it appears:**
- `setup_data_tracking.sh` — initialises DVC and adds the data directory
- `requirements.txt` — listed as a dependency

### joblib

**What it is:** A library for lightweight pipelining in Python, commonly used to serialise (save) and deserialise (load) Python objects, especially scikit-learn models.

**Why it is used:** `evaluate.py` loads a model from a `.joblib` file. This is a simpler, local alternative to loading from MLflow.

**Where it appears:**
- `evaluate.py` — `joblib.load(model_path)`

### python-dotenv

**What it is:** A library that reads key-value pairs from a `.env` file and loads them into environment variables.

**Why it is used:** Keeping secrets (Databricks tokens) and environment-specific settings out of source code. Scripts call `load_dotenv()` at the top, which reads `.env` before any other code runs.

**Where it appears:**
- `train.py` — `load_dotenv()` at module level
- `predict_remote.py` — `load_dotenv()` at module level

### pytest

**What it is:** The standard Python testing framework.

**Why it is used:** Running the automated test suite. pytest discovers test files automatically and provides the `monkeypatch` fixture used to stub out MLflow calls in `test_api.py`.

**Where it appears:**
- `tests/test_api.py`, `tests/test_data.py`
- `requirements.txt`
- `.github/workflows/ci.yml` — `python -m pytest -q`

## Infrastructure Tools

### Docker

**What it is:** A platform for building and running containers. A container packages an application and all its dependencies into a single, portable unit that runs identically on any machine.

**Why it is used:** Ensuring that the API runs the same way in development, CI, and production. The Docker image bakes in the trained model, so no MLflow connection is needed at container startup when using the local SQLite tracking store.

**Where it appears:**
- `Dockerfile` — build instructions
- `.dockerignore` — files excluded from the build context
- `.github/workflows/ci.yml` — `docker/build-push-action`

### GitHub Actions

**What it is:** GitHub's built-in CI/CD (Continuous Integration / Continuous Delivery) platform. Workflows are defined as YAML files in `.github/workflows/` and triggered by Git events.

**Why it is used:** Automatically running tests on every pull request (preventing broken code from being merged) and automatically publishing the Docker image on every merge to `main`.

**Where it appears:**
- `.github/workflows/ci.yml`

### GitHub Container Registry (GHCR)

**What it is:** A Docker image registry hosted by GitHub. Images are stored at `ghcr.io/<owner>/<image-name>`.

**Why it is used:** Storing the published Docker image in the same place as the source code, with access controlled by GitHub permissions.

**Where it appears:**
- `.github/workflows/ci.yml` — the `build-and-push` job

---

Previous: [Project Structure ←](03-project-structure.md) | Next: [Configuration →](05-configuration.md)
