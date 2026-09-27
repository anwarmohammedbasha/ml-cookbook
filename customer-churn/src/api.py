"""
FastAPI prediction service for the customer-churn project.

Loads the most recent finished model from the configured MLflow experiment
at startup and exposes a single POST /predict endpoint.  The model is
stored in application state for the lifetime of the server process,
avoiding the cost of re-loading it on every request.

Usage
-----
    uvicorn --env-file .env src.api:app --host 0.0.0.0 --port 8000

Environment
-----------
    MLFLOW_TRACKING_URI          Defaults to the local SQLite file.
    DATABRICKS_EXPERIMENT_PATH   Defaults to "customer-churn".
    DATABRICKS_HOST / DATABRICKS_TOKEN
                                 Required when using Databricks tracking.
"""

import logging
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from src.constants import FEATURE_COLUMNS, MODEL_ARTIFACT_NAME, PROJECT_ROOT
from src.utils import get_latest_run_id

logger = logging.getLogger(__name__)

# Read from the environment so the API uses the same experiment as the
# training script without requiring a code change between environments.
EXPERIMENT_NAME: str = os.environ.get(
    "DATABRICKS_EXPERIMENT_PATH", "customer-churn"
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Input features for a single churn prediction request."""

    # account_age must be a positive integer (gt=0 excludes zero).
    account_age: int = Field(gt=0, description="Months since the customer subscribed.")
    monthly_spend: float = Field(ge=0, description="Monthly subscription spend in USD.")
    support_tickets: int = Field(ge=0, description="Number of support tickets raised.")


class ChurnPrediction(BaseModel):
    """Churn prediction response returned by POST /predict."""

    churn_probability: float = Field(
        ge=0, le=1, description="Model confidence that the customer will churn."
    )
    churn_label: int = Field(
        ge=0, le=1, description="Binary prediction: 1 = churn, 0 = retain."
    )


# ---------------------------------------------------------------------------
# Application lifespan — model loading
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the latest trained model from MLflow before accepting requests.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown.  Raising an exception before ``yield`` prevents the server
    from starting, which is the correct behaviour when no model is
    available — a server with no model cannot serve predictions.
    """
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(
        "Connecting to MLflow tracking store: %s | experiment: %s",
        tracking_uri,
        EXPERIMENT_NAME,
    )

    run_id = get_latest_run_id(EXPERIMENT_NAME)
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"

    logger.info("Loading model from %s", model_uri)
    # Store the model on app.state so every request handler can access it
    # via request.app.state.model without a global variable.
    app.state.model = mlflow.sklearn.load_model(model_uri)
    logger.info("Model loaded successfully. Server is ready.")

    yield
    # No explicit shutdown logic required; the model is garbage-collected
    # when the process exits.


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a SaaS customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures, request: Request) -> ChurnPrediction:
    """Return a churn prediction for a single customer.

    FastAPI validates the request body against ``CustomerFeatures`` before
    this function is called.  Invalid requests receive a 422 response
    automatically.

    Parameters
    ----------
    customer:
        Validated customer feature values from the JSON request body.
    request:
        The raw Starlette request, used to access ``app.state.model``.

    Returns
    -------
    ChurnPrediction
        ``churn_probability`` (float 0–1) and ``churn_label`` (0 or 1).
    """
    model = request.app.state.model

    # Build a single-row DataFrame with columns in the exact order the
    # model was trained on.  Column order matters for sklearn pipelines.
    features = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)

    churn_label = int(model.predict(features)[0])

    # predict_proba returns a 2-D array: rows = samples, columns = classes.
    # model.classes_ maps column indices to class labels; we need the
    # column for class 1 (churn) regardless of the internal ordering.
    positive_class_index = list(model.classes_).index(1)
    churn_probability = float(model.predict_proba(features)[0][positive_class_index])

    return ChurnPrediction(
        churn_probability=churn_probability,
        churn_label=churn_label,
    )
