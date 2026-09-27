"""Tests for the FastAPI prediction endpoint."""

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api


class DummyModel:
    """Minimal sklearn-compatible model stub used across API tests."""

    classes_ = [0, 1]

    def predict(self, features):
        return [1]

    def predict_proba(self, features):
        return [[0.2, 0.8]]


# ---------------------------------------------------------------------------
# Shared monkeypatching helpers
# ---------------------------------------------------------------------------


def _patch_mlflow_happy_path(monkeypatch) -> None:
    """Replace all MLflow calls with stubs that simulate a healthy setup."""
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        api.mlflow,
        "get_experiment_by_name",
        lambda name: SimpleNamespace(experiment_id="test-experiment"),
    )
    monkeypatch.setattr(
        api.mlflow,
        "search_runs",
        lambda **kwargs: pd.DataFrame([{"run_id": "test-run"}]),
    )
    monkeypatch.setattr(
        api.mlflow.sklearn, "load_model", lambda model_uri: DummyModel()
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_predict_returns_churn_probability_and_label(monkeypatch) -> None:
    """A valid request must return HTTP 200 with the expected JSON body."""
    _patch_mlflow_happy_path(monkeypatch)

    with TestClient(api.app) as client:
        response = client.post(
            "/predict",
            json={"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3},
        )

    assert response.status_code == 200
    assert response.json() == {"churn_probability": 0.8, "churn_label": 1}


# ---------------------------------------------------------------------------
# Input validation tests (Pydantic / FastAPI 422 responses)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"account_age": 0, "monthly_spend": 89.5, "support_tickets": 3},   # age = 0 violates gt=0
        {"account_age": -5, "monthly_spend": 89.5, "support_tickets": 3},  # negative age
        {"account_age": 12, "monthly_spend": -1.0, "support_tickets": 3},  # negative spend
        {"account_age": 12, "monthly_spend": 89.5, "support_tickets": -1}, # negative tickets
        {"monthly_spend": 89.5, "support_tickets": 3},                     # missing account_age
    ],
)
def test_predict_rejects_invalid_input(monkeypatch, payload: dict) -> None:
    """Invalid request bodies must be rejected with HTTP 422."""
    _patch_mlflow_happy_path(monkeypatch)

    with TestClient(api.app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Startup failure tests
# ---------------------------------------------------------------------------


def test_startup_fails_when_experiment_not_found(monkeypatch) -> None:
    """Server startup must raise RuntimeError if the experiment is missing."""
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        api.mlflow, "get_experiment_by_name", lambda name: None
    )

    with pytest.raises(RuntimeError, match="was not found"):
        with TestClient(api.app):
            pass


def test_startup_fails_when_no_finished_runs(monkeypatch) -> None:
    """Server startup must raise RuntimeError if no finished runs exist."""
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(
        api.mlflow,
        "get_experiment_by_name",
        lambda name: SimpleNamespace(experiment_id="test-experiment"),
    )
    monkeypatch.setattr(
        api.mlflow,
        "search_runs",
        lambda **kwargs: pd.DataFrame(),  # empty — no runs
    )

    with pytest.raises(RuntimeError, match="No completed runs"):
        with TestClient(api.app):
            pass
