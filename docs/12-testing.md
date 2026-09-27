# 12 — Testing

**Source files:**
- [`customer-churn/tests/test_data.py`](../customer-churn/tests/test_data.py)
- [`customer-churn/tests/test_api.py`](../customer-churn/tests/test_api.py)

## Overview

The test suite has two files covering the two most important components: data generation and the prediction API. Tests are written with pytest and run without any external dependencies (no MLflow server, no Databricks connection, no pre-existing data files).

## `test_data.py` — Data Generation Tests

```python
from src.data_processor import FEATURE_COLUMNS, TARGET_COLUMN, generate_data

def test_generated_dataset_has_expected_columns_and_no_nulls():
    dataset = generate_data()

    assert len(dataset) == 1000
    assert list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert dataset.notna().all().all()
```

### What This Test Verifies

| Assertion | What It Checks |
|-----------|---------------|
| `len(dataset) == 1000` | The default `n_rows=1000` produces exactly 1,000 rows |
| `list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]` | Columns are `["account_age", "monthly_spend", "support_tickets", "churn"]` in the correct order |
| `dataset.notna().all().all()` | No cell in the DataFrame is `NaN` or `None` |

`dataset.notna()` returns a boolean DataFrame (True where values are not null). `.all()` reduces each column to a single True/False. The second `.all()` reduces the resulting Series to a single True/False.

### Why This Test Matters

If `generate_data()` is ever modified — for example, to add a new feature or change the distribution — this test will catch regressions in the output shape and data quality.

## `test_api.py` — API Endpoint Tests

```python
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
            json={"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3},
        )

    assert response.status_code == 200
    assert response.json() == {"churn_probability": 0.8, "churn_label": 1}
```

### The `DummyModel` Class

`DummyModel` is a test double (a fake object) that mimics the interface of a `RandomForestClassifier` without actually being one. It implements the three attributes/methods that `api.py` uses:

| Attribute/Method | Real Behaviour | Dummy Behaviour |
|-----------------|----------------|-----------------|
| `classes_` | Array of class labels from training | `[0, 1]` |
| `predict(features)` | Returns majority-vote predictions | Always returns `[1]` |
| `predict_proba(features)` | Returns class probabilities | Always returns `[[0.2, 0.8]]` |

### `monkeypatch` — Replacing MLflow Calls

`monkeypatch` is a pytest fixture that temporarily replaces attributes on objects for the duration of a test. Here it replaces four MLflow functions:

| Replaced Function | Replacement | Why |
|------------------|-------------|-----|
| `mlflow.set_tracking_uri` | No-op lambda | Prevents connecting to any MLflow server |
| `mlflow.get_experiment_by_name` | Returns a fake experiment | Avoids needing a real experiment |
| `mlflow.search_runs` | Returns a DataFrame with a fake run ID | Avoids needing real run history |
| `mlflow.sklearn.load_model` | Returns `DummyModel()` | Avoids downloading a real model |

All replacements are automatically reverted after the test completes.

### `TestClient` — Testing FastAPI Without a Server

`fastapi.testclient.TestClient` wraps the FastAPI app and lets you make HTTP requests directly in Python without starting a real server. Using it as a context manager (`with TestClient(api.app) as client`) triggers the lifespan startup and shutdown events, which is where the model is loaded.

### What This Test Verifies

1. The lifespan startup completes without error (model is loaded into `app.state`).
2. A valid POST request to `/predict` returns HTTP 200.
3. The response body matches the expected JSON exactly.

## Running the Tests

```bash
cd customer-churn
python -m pytest -q
```

Expected output:
```
..
2 passed in X.XXs
```

The `-q` flag suppresses verbose output. Remove it for more detail:

```bash
python -m pytest -v
```

## Test Coverage Gaps

The current test suite does not cover:

- **Invalid input validation** — e.g., sending `account_age: -1` should return HTTP 422.
- **Missing experiment** — the lifespan should raise `RuntimeError` if the experiment does not exist.
- **Empty run history** — the lifespan should raise `RuntimeError` if no completed runs exist.
- **`data_processor.process_data()`** — the file I/O path is not tested.
- **`train.py`** — training is not tested (would require a real or mocked MLflow server).
- **`evaluate.py`** — not tested.
- **`predict_remote.py`** — not tested.

---

Previous: [Data Versioning with DVC ←](11-data-versioning.md) | Next: [Docker and Containerisation →](13-docker.md)
