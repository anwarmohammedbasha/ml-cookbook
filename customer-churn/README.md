# Customer Churn MLOps Blueprint

A production-structured MLOps blueprint that predicts SaaS customer churn. It generates reproducible synthetic data, trains and evaluates a Random Forest, tracks experiments and model artifacts with MLflow, versions processed data with DVC, and serves predictions over a FastAPI REST API.

## Architecture and Stack

| Component | Implementation |
| --- | --- |
| Data preparation | pandas, NumPy, scikit-learn `train_test_split` |
| Model | Scikit-Learn `RandomForestClassifier` |
| Experiment tracking | MLflow — Databricks or local SQLite |
| Data versioning | DVC pointer files in Git; data in DVC cache/remote |
| Serving | FastAPI with Pydantic request validation |
| Packaging | Docker, based on `python:3.11-slim` |
| Automation | GitHub Actions — tests on PRs, publishes image to GHCR on `main` |

Model features: `account_age`, `monthly_spend`, `support_tickets`. Target: `churn`.

## Project Structure

```
customer-churn/
├── configs/
│   └── config.yaml          # Hyperparameters and data paths
├── src/
│   ├── __init__.py
│   ├── constants.py         # Shared constants (paths, column names, artifact keys)
│   ├── utils.py             # Shared helpers: config loading, MLflow run discovery
│   ├── data_processor.py    # Synthetic data generation and train/test splitting
│   ├── train.py             # Model training and MLflow experiment logging
│   ├── evaluate.py          # Offline evaluation against the test set
│   ├── api.py               # FastAPI prediction service
│   └── predict_remote.py    # CLI smoke-test for remote MLflow deployments
├── tests/
│   ├── test_data.py         # 9 tests covering data generation
│   └── test_api.py          # 8 tests covering the prediction endpoint
├── .env.example             # Environment variable template
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── setup_data_tracking.sh
```

## Local Setup

Run all commands from the `customer-churn/` directory.

**1. Install dependencies (Python 3.11+):**

```bash
python -m pip install -r requirements.txt
```

**2. Create your environment file:**

```bash
cp .env.example .env
```

**3. Fill in `.env`:**

For Databricks Community Edition:

```dotenv
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://<your-workspace-host>
DATABRICKS_TOKEN=<your-workspace-access-token>
DATABRICKS_EXPERIMENT_PATH=/Users/<your-email>/customer-churn
```

For local tracking (no account needed — omit the Databricks variables):

```dotenv
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
DATABRICKS_EXPERIMENT_PATH=customer-churn
```

Never commit `.env` or share its token.

## Run the Pipeline

All scripts require `PYTHONPATH=.` (or equivalent) so the `src` package resolves correctly. The examples below use the inline `env` syntax; alternatively, `export PYTHONPATH=.` once in your shell.

**Generate data:**

```bash
PYTHONPATH=. python src/data_processor.py
# Windows: set PYTHONPATH=. && python src/data_processor.py
```

**Train the model:**

```bash
PYTHONPATH=. python src/train.py
```

Logs hyperparameters, accuracy, precision, recall, and the model artifact to the configured MLflow experiment.

**Evaluate the latest model:**

```bash
PYTHONPATH=. python src/evaluate.py
```

Loads the most recent finished run from MLflow and prints a full classification report.

**Remote smoke-test (Databricks only):**

```bash
PYTHONPATH=. python src/predict_remote.py
```

**Run the tests:**

```bash
python -m pytest -v
```

`pytest.ini` sets `PYTHONPATH=.` automatically via `pythonpath = .`, so no prefix is needed for tests.

## Serve the REST API

Run training first, then start the server:

```bash
PYTHONPATH=. uvicorn --env-file .env src.api:app --host 0.0.0.0 --port 8000
```

**Make a prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}'
```

**Response:**

```json
{"churn_probability": 0.35, "churn_label": 0}
```

Interactive API docs: `http://localhost:8000/docs`

## Version Data with DVC

```bash
PYTHONPATH=. python src/data_processor.py
bash setup_data_tracking.sh
```

Then commit the DVC metadata and push data to your remote:

```bash
dvc remote add -d storage <dvc-remote-url>
dvc push
```

## Docker and CI/CD

Build and run the container:

```bash
docker build -t customer-churn-api .
docker run --rm --env-file .env -p 8000:8000 customer-churn-api
```

The Docker build generates data and trains a model inside the image (`PYTHONPATH=/app` is set in the `ENV` layer). The GitHub Actions workflow runs pytest on every push and pull request to `main`, and builds and pushes `ghcr.io/<owner>/customer-churn-api:latest` on merges to `main`.

## Configuration Reference

`configs/config.yaml` controls all pipeline parameters:

```yaml
data:
  raw_path: data/raw/customers.csv
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv

model:
  n_estimators: 200
  max_depth: 8
  random_state: 42

split:
  test_size: 0.2
  random_state: 42
```

All hyperparameters are logged to MLflow automatically on each training run.
