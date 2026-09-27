from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "random_forest.joblib"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]


class CustomerFeatures(BaseModel):
    account_age: int = Field(gt=0)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)


class ChurnPrediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_label: int = Field(ge=0, le=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
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