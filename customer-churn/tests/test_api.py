from fastapi.testclient import TestClient

from src import api


class DummyModel:
    classes_ = [0, 1]

    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


def test_predict_returns_churn_probability_and_label(monkeypatch):
    monkeypatch.setattr(api.joblib, "load", lambda model_path: DummyModel())

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