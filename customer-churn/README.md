# Customer Churn MLOps Blueprint

This project builds a binary classifier for predicting whether a SaaS customer will churn. It generates reproducible synthetic customer data, trains and evaluates a Random Forest, tracks experiments and model artifacts with MLflow, versions processed data with DVC, and serves predictions over a FastAPI REST API.

## Architecture and Stack

| Component | Implementation |
| --- | --- |
| Data preparation | pandas, NumPy, scikit-learn train/test split |
| Model | Scikit-Learn `RandomForestClassifier` |
| Experiment tracking | MLflow, configured for Databricks or a local SQLite tracking store |
| Data versioning | DVC pointer files committed to Git; data stored by DVC |
| Serving | FastAPI with Pydantic request validation |
| Packaging | Docker, based on `python:3.11-slim` |
| Automation | GitHub Actions tests pull requests and publishes the image to GHCR from `main` |

The model features are `account_age`, `monthly_spend`, and `support_tickets`; the target is `churn`. The configured experiment is selected by `DATABRICKS_EXPERIMENT_PATH` and defaults to `customer-churn`.

## Local Setup

Run commands from the `customer-churn` directory. Use Python 3.11 or later and install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a local environment file from the template:

```bash
cp .env.example .env
```

For Databricks Community Edition, set the workspace base URL in `DATABRICKS_HOST`. Use only the scheme and host from the browser URL; do not include a path or query string. Set `MLFLOW_TRACKING_URI=databricks` and provide a workspace access token. Never commit `.env` or paste its token into logs, source control, or support messages.

```dotenv
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://<your-workspace-host>
DATABRICKS_TOKEN=<your-workspace-access-token>
DATABRICKS_EXPERIMENT_PATH=/Users/<your-email>/customer-churn

# Optional generic MLflow REST bearer-token setting. The Databricks SDK
# authentication used by this project reads DATABRICKS_TOKEN instead.
MLFLOW_TRACKING_TOKEN=<your-mlflow-rest-token>
```

`MLFLOW_TRACKING_TOKEN` is not a replacement for `DATABRICKS_TOKEN` in this project's Databricks SDK configuration. If you are using a self-hosted MLflow server with HTTP token authentication instead, follow that server's authentication configuration. The tracking URI and credentials are loaded from the environment by the training and remote-prediction scripts.

## Run the Pipeline

Generate 1,000 synthetic records and write raw and processed CSVs:

```bash
python src/data_processor.py
```

Train on the processed training split. The script evaluates on the test split and logs hyperparameters, accuracy, precision, recall, and the model artifact to the configured MLflow experiment:

```bash
python src/train.py
```

Download the latest completed model artifact from MLflow and print a prediction for a sample customer:

```bash
python src/predict_remote.py
```

Run the tests from the project directory:

```bash
python -m pytest -q
```

## Serve the REST API

The API loads the latest completed model from the configured MLflow experiment at startup. Run training first, then launch Uvicorn with the project environment file:

```bash
uvicorn --env-file .env src.api:app --host 0.0.0.0 --port 8000
```

Submit a prediction request:

```bash
curl -X POST http://localhost:8000/predict \
	-H 'Content-Type: application/json' \
	-d '{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}'
```

The response contains `churn_probability` and `churn_label`. Interactive API documentation is available at `http://localhost:8000/docs`.

## Version Data with DVC

Generate processed data first, then initialize DVC and create the pointer file:

```bash
python src/data_processor.py
bash setup_data_tracking.sh
```

The script prints the Git commands for committing DVC metadata. Configure a DVC remote before sharing data, then upload the cached dataset:

```bash
dvc remote add -d storage <dvc-remote-url>
dvc push
```

Commit the remote configuration only if it contains no credentials. Store remote credentials using DVC-supported environment variables or your team's secret manager.

## Docker and CI/CD

Build and start the container from this directory:

```bash
docker build -t customer-churn-api .
docker run --rm --env-file .env -p 8000:8000 customer-churn-api
```

The Docker build generates data and trains a model inside the image. The GitHub Actions workflow runs pytest for pushes and pull requests to `main` and builds/pushes `ghcr.io/<owner>/customer-churn-api:latest` on pushes to `main`.