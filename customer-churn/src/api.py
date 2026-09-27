from contextlib import asynccontextmanager
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
EXPERIMENT_NAME = "customer-churn"
MODEL_ARTIFACT_NAME = "random_forest"


class CustomerFeatures(BaseModel):
    account_age: int = Field(gt=0)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class ChurnPrediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_label: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment '{EXPERIMENT_NAME}' was not found.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(f"No completed runs found for '{EXPERIMENT_NAME}'.")

    run_id = runs.iloc[0]["run_id"]
    app.state.model = mlflow.sklearn.load_model(
        f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    )
    yield


app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)


@app.post("/predict", response_model=ChurnPrediction)
def predict(customer: CustomerFeatures, request: Request) -> ChurnPrediction:
    model = request.app.state.model
    features = pd.DataFrame([customer.model_dump()], columns=FEATURE_COLUMNS)
    churn_label = int(model.predict(features)[0])
    positive_class_index = list(model.classes_).index(1)
    churn_probability = float(model.predict_proba(features)[0][positive_class_index])

    return ChurnPrediction(
        churn_probability=churn_probability,
        churn_label=churn_label,
    )