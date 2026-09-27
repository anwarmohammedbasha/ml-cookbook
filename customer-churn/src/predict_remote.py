"""
Remote prediction verification script for the customer-churn project.

Connects to the configured MLflow tracking store, downloads the most
recent finished model, and runs a prediction for a sample customer.
Use this script to verify that a remote Databricks deployment is working
end-to-end after a training run.

Usage
-----
    python src/predict_remote.py

Environment
-----------
    MLFLOW_TRACKING_URI      Must be set (no local fallback — this script
                             is intentionally for remote verification).
    DATABRICKS_TOKEN         Must be set when using Databricks tracking.
    DATABRICKS_EXPERIMENT_PATH
                             Defaults to "customer-churn".
    DATABRICKS_HOST          Required when MLFLOW_TRACKING_URI=databricks.
"""

# load_dotenv() must precede any library that reads os.environ at import
# time (e.g. the Databricks SDK used by mlflow).
from dotenv import load_dotenv

load_dotenv()

import logging
import os

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.constants import FEATURE_COLUMNS, MODEL_ARTIFACT_NAME
from src.utils import get_latest_run_id

logger = logging.getLogger(__name__)

EXPERIMENT_NAME: str = os.getenv("DATABRICKS_EXPERIMENT_PATH", "customer-churn")


def predict_latest_run() -> int:
    """Load the latest remote model and predict churn for a sample customer.

    This function is a smoke-test for the remote MLflow deployment.  It
    intentionally requires both ``MLFLOW_TRACKING_URI`` and
    ``DATABRICKS_TOKEN`` to be set — unlike the API, which falls back to
    a local SQLite store — because its purpose is to verify remote
    connectivity.

    Returns
    -------
    int
        The binary churn prediction (0 or 1) for the sample customer.

    Raises
    ------
    RuntimeError
        If ``MLFLOW_TRACKING_URI`` or ``DATABRICKS_TOKEN`` is not set, or
        if no finished run exists for the configured experiment.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI must be set in the environment or .env."
        )
    if not os.environ.get("DATABRICKS_TOKEN"):
        raise RuntimeError(
            "DATABRICKS_TOKEN must be set in the environment or .env."
        )

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s | experiment: %s", tracking_uri, EXPERIMENT_NAME)

    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"

    logger.info("Loading model from %s", model_uri)
    # pyfunc is used here (rather than mlflow.sklearn) because this script
    # only needs a binary prediction, not sklearn-specific methods like
    # predict_proba() or classes_.  pyfunc works for any MLflow model
    # flavour, making the script more portable.
    model = mlflow.pyfunc.load_model(model_uri)

    # Sample customer matching the README curl example so the output can
    # be compared directly with the API response.
    sample_customer = pd.DataFrame(
        [{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(sample_customer)
    result = int(prediction[0])

    logger.info("Prediction for run %s: %d", run_id, result)
    print(f"Churn prediction for run {run_id}: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    predict_latest_run()
