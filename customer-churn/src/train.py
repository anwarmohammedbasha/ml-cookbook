"""
Model training script for the customer-churn project.

Reads the processed train/test CSVs, trains a Random Forest classifier,
evaluates it on the held-out test set, and logs all parameters, metrics,
and the serialised model artifact to the configured MLflow experiment.

Usage
-----
    python src/train.py

Environment
-----------
    MLFLOW_TRACKING_URI      Where MLflow stores run data.
                             Defaults to a local SQLite file (mlflow.db).
    DATABRICKS_EXPERIMENT_PATH
                             MLflow experiment name or Databricks path.
                             Defaults to "customer-churn".
    DATABRICKS_HOST          Required when MLFLOW_TRACKING_URI=databricks.
    DATABRICKS_TOKEN         Required when MLFLOW_TRACKING_URI=databricks.
"""

# load_dotenv() must be called before any library that reads os.environ
# at import time (e.g. the Databricks SDK inside mlflow).
from dotenv import load_dotenv

load_dotenv()

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.constants import (
    CONFIG_PATH,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_NAME,
    PROJECT_ROOT,
    TARGET_COLUMN,
)
from src.utils import load_config

logger = logging.getLogger(__name__)

# Read experiment name from the environment so the same script works for
# both local SQLite tracking and remote Databricks tracking without code
# changes.  The default matches the experiment created by the Dockerfile.
EXPERIMENT_NAME: str = os.getenv("DATABRICKS_EXPERIMENT_PATH", "customer-churn")


def train() -> None:
    """Train a Random Forest classifier and log the run to MLflow.

    Reads hyperparameters and data paths from ``configs/config.yaml``.
    All parameters, evaluation metrics (accuracy, precision, recall), and
    the serialised model artifact are recorded under a single MLflow run.
    The run is automatically marked FINISHED on success or FAILED if an
    unhandled exception propagates out of the context manager.

    Raises
    ------
    FileNotFoundError
        If the processed train or test CSV does not exist.  Run
        ``data_processor.py`` first.
    mlflow.exceptions.MlflowException
        If the MLflow tracking server is unreachable or authentication
        fails.
    """
    config = load_config()

    train_path = PROJECT_ROOT / config["data"]["train_path"]
    test_path = PROJECT_ROOT / config["data"]["test_path"]

    logger.info("Loading training data from %s", train_path)
    train_data = pd.read_csv(train_path)

    logger.info("Loading test data from %s", test_path)
    test_data = pd.read_csv(test_path)

    # Fall back to a local SQLite database when no tracking URI is set,
    # so the script works out-of-the-box without any external services.
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        model_params = {
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "random_state": config["model"]["random_state"],
        }
        mlflow.log_params(model_params)
        logger.info("Training RandomForestClassifier with params: %s", model_params)

        model = RandomForestClassifier(**model_params)
        model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])

        predictions = model.predict(test_data[FEATURE_COLUMNS])
        metrics = {
            "accuracy": accuracy_score(test_data[TARGET_COLUMN], predictions),
            # zero_division=0 avoids a warning when the model predicts no
            # positive cases, which can happen on very small datasets.
            "precision": precision_score(
                test_data[TARGET_COLUMN], predictions, zero_division=0
            ),
            "recall": recall_score(
                test_data[TARGET_COLUMN], predictions, zero_division=0
            ),
        }
        mlflow.log_metrics(metrics)
        logger.info("Evaluation metrics: %s", metrics)

        # skops_trusted_types explicitly whitelists the internal Cython
        # tree type so the model can be deserialised safely without
        # allowing arbitrary pickle execution.
        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        logger.info("Logged model artifact '%s' to run %s", MODEL_ARTIFACT_NAME, run.info.run_id)
        print(f"Logged model to MLflow run {run.info.run_id}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
