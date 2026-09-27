# 10 — Remote Prediction

**Source file:** [`customer-churn/src/predict_remote.py`](../customer-churn/src/predict_remote.py)

## Responsibilities

`predict_remote.py` is a command-line script that:

1. Connects to the configured MLflow tracking store (typically Databricks).
2. Finds the most recent completed training run.
3. Downloads the model artifact.
4. Runs a single prediction for a hardcoded sample customer.
5. Prints the result.

It is primarily a **verification tool** — a quick way to confirm that a remote MLflow deployment is working end-to-end.

## Difference from `api.py`

Both `predict_remote.py` and `api.py` load a model from MLflow and run a prediction. The key differences are:

| Aspect | `predict_remote.py` | `api.py` |
|--------|---------------------|----------|
| Purpose | One-off CLI verification | Long-running HTTP service |
| Model loading | `mlflow.pyfunc.load_model()` | `mlflow.sklearn.load_model()` |
| Input | Hardcoded sample customer | HTTP request body |
| Output | Printed to stdout | JSON HTTP response |
| Requires Databricks | Yes (validates `DATABRICKS_TOKEN`) | No (falls back to SQLite) |

## `mlflow.pyfunc` vs. `mlflow.sklearn`

`predict_remote.py` uses `mlflow.pyfunc.load_model()` while `api.py` uses `mlflow.sklearn.load_model()`.

- **`mlflow.pyfunc`** is the generic MLflow model interface. It loads any MLflow-logged model as a `PythonModel` with a `.predict(dataframe)` method. This is the most portable approach — it works regardless of the underlying framework.
- **`mlflow.sklearn`** loads the model as a native scikit-learn object, giving access to scikit-learn-specific methods like `.predict_proba()` and `.classes_`. The API needs these for probability output, which is why it uses the sklearn flavour.

## The `predict_latest_run()` Function

### Step 1 — Validate Environment

```python
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if not tracking_uri:
    raise RuntimeError("MLFLOW_TRACKING_URI must be set in the environment or .env.")
if not os.environ.get("DATABRICKS_TOKEN"):
    raise RuntimeError("DATABRICKS_TOKEN must be set in the environment or .env.")
```

Unlike `train.py` and `api.py`, this script **requires** both variables to be set. It does not fall back to a local SQLite database. This is intentional — the script's purpose is to verify a remote connection.

### Step 2 — Find the Latest Run

```python
mlflow.set_tracking_uri(tracking_uri)
experiment = mlflow.get_experiment_by_name(DATABRICKS_EXPERIMENT_PATH)
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
run_id = runs.iloc[0]["run_id"]
```

This is the same run-discovery logic used in `api.py`. It finds the most recently started run with a `FINISHED` status.

### Step 3 — Load and Predict

```python
model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
model = mlflow.pyfunc.load_model(model_uri)

customer = pd.DataFrame(
    [{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}],
    columns=FEATURE_COLUMNS,
)
prediction = model.predict(customer)
print(f"Churn prediction for run {run_id}: {prediction[0]}")
```

The sample customer (12 months old, $89.50/month, 3 support tickets) is the same one used in the README's `curl` example, making it easy to compare the script output with the API response.

## Execution

```bash
cd customer-churn
python src/predict_remote.py
# Output: Churn prediction for run <run_id>: 1
```

---

Previous: [Prediction API ←](09-prediction-api.md) | Next: [Data Versioning with DVC →](11-data-versioning.md)
