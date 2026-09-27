from types import SimpleNamespace

import pandas as pd
from fastapi.testclient import TestClient

from src import api


class DummyModel:
    classes_ = [0, 1]

    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


def test_predict_returns_churn_probability_and_label(monkeypatch):
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda tracking_uri: None)
    monkeypatch.setattr(
        api.mlflow,
        "get_experiment_by_name",
        lambda experiment_name: SimpleNamespace(experiment_id="test-experiment"),
    )
    monkeypatch.setattr(
        api.mlflow,
        "search_runs",
        lambda **kwargs: pd.DataFrame([{"run_id": "test-run"}]),
    )
    monkeypatch.setattr(
        api.mlflow.sklearn, "load_model", lambda model_uri: DummyModel()
    )

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={
                "account_age": 12,
                "monthly_spend": 89.5,
                "support_tickets": 3,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"churn_probability": 0.8, "churn_label": 1}