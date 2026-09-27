"""
Standalone model evaluation script for the customer-churn project.

Loads the most recent finished model from the configured MLflow experiment
and prints a full classification report against the held-out test set.
This script is intentionally independent of the API server and is useful
for quick offline evaluation after a training run.

Usage
-----
    python src/evaluate.py

Environment
-----------
    MLFLOW_TRACKING_URI      Defaults to the local SQLite file (mlflow.db).
    DATABRICKS_EXPERIMENT_PATH
                             Defaults to "customer-churn".
    DATABRICKS_HOST / DATABRICKS_TOKEN
                             Required when using Databricks tracking.
"""

# load_dotenv() must precede any library that reads os.environ at import
# time (e.g. the Databricks SDK used by mlflow).
from dotenv import load_dotenv

load_dotenv()

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report

from src.constants import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN
from src.utils import get_latest_run_id, load_config

logger = logging.getLogger(__name__)

EXPERIMENT_NAME: str = os.getenv("DATABRICKS_EXPERIMENT_PATH", "customer-churn")


def evaluate() -> None:
    """Evaluate the latest MLflow model against the test set.

    Connects to the configured MLflow tracking store, retrieves the most
    recent finished run, loads its model artifact, and prints a
    per-class classification report (precision, recall, F1, support).

    Raises
    ------
    FileNotFoundError
        If the test CSV does not exist.  Run ``data_processor.py`` first.
    RuntimeError
        If no finished MLflow run exists for the configured experiment.
    """
    config = load_config()
    test_path = PROJECT_ROOT / config["data"]["test_path"]

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, EXPERIMENT_NAME)

    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/random_forest"

    logger.info("Loading model from %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    logger.info("Loading test data from %s", test_path)
    test_data = pd.read_csv(test_path)

    predictions = model.predict(test_data[FEATURE_COLUMNS])
    report = classification_report(test_data[TARGET_COLUMN], predictions)
    print(report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    evaluate()
