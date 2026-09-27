# 09 — Prediction API

**Source file:** [`customer-churn/src/api.py`](../customer-churn/src/api.py)

## Responsibilities

`api.py` is the main deliverable of the project. It is a FastAPI application that:

1. Loads the latest trained model from MLflow at startup.
2. Exposes a `POST /predict` endpoint that accepts customer features and returns a churn prediction.

## Pydantic Schemas

FastAPI uses Pydantic models to define the shape and validation rules for request and response bodies.

### `CustomerFeatures` — Request Schema

```python
class CustomerFeatures(BaseModel):
    account_age: int = Field(gt=0)
    monthly_spend: float = Field(ge=0)
    support_tickets: int = Field(ge=0)
```

| Field | Type | Constraint | Meaning |
|-------|------|-----------|---------|
| `account_age` | `int` | `gt=0` (greater than 0) | Must be a positive integer |
| `monthly_spend` | `float` | `ge=0` (greater than or equal to 0) | Cannot be negative |
| `support_tickets` | `int` | `ge=0` | Cannot be negative |

If a request violates any constraint, FastAPI automatically returns a `422 Unprocessable Entity` response with a detailed error message — no validation code needs to be written manually.

### `ChurnPrediction` — Response Schema

```python
class ChurnPrediction(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    churn_label: int = Field(ge=0, le=1)
```

| Field | Type | Constraint | Meaning |
|-------|------|-----------|---------|
| `churn_probability` | `float` | `0 ≤ x ≤ 1` | Probability that the customer will churn |
| `churn_label` | `int` | `0` or `1` | Binary prediction: `1` = churn, `0` = stay |

## ASGI and the Async Context Manager

**ASGI** (Asynchronous Server Gateway Interface) is the protocol that connects an async Python web framework (FastAPI) to an async web server (Uvicorn). Unlike the older WSGI protocol, ASGI supports long-lived connections and concurrent request handling without threads.

FastAPI is built on top of **Starlette**, which is an ASGI framework. When Uvicorn starts, it calls the ASGI application's lifespan handler to signal startup and shutdown events.

The `@asynccontextmanager` decorator from Python's `contextlib` module converts an `async` generator function into an async context manager. The code before `yield` runs on startup; the code after `yield` (if any) runs on shutdown. This is the modern FastAPI pattern for startup/shutdown logic, replacing the older `@app.on_event("startup")` decorator.

## Application Lifespan — Model Loading at Startup

```python
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
```

### How the Lifespan Context Manager Works

The `lifespan` function is an **async context manager** registered with the FastAPI app. FastAPI calls everything before `yield` during startup and everything after `yield` during shutdown.

**Startup sequence:**
1. Read `MLFLOW_TRACKING_URI` from the environment (falls back to local SQLite).
2. Look up the experiment named `"customer-churn"`.
3. Search for the most recent completed run in that experiment.
4. Load the model artifact from that run.
5. Store the model in `app.state.model` — a dictionary attached to the app instance that persists for the lifetime of the server.
6. `yield` — the server is now ready to accept requests.

**Why load at startup?** Loading the model once at startup is far more efficient than loading it on every request. A Random Forest with 200 trees takes a non-trivial amount of time to deserialise from MLflow.

**Why `app.state`?** FastAPI's `app.state` is a simple namespace for storing application-level state. It is accessible from any request handler via `request.app.state`.

**What happens if no model is found?** The `RuntimeError` is raised before `yield`, which causes the server to fail to start. This is the correct behaviour — a server with no model cannot serve predictions.

## The `app` Instance

```python
app = FastAPI(title="Customer Churn Prediction API", lifespan=lifespan)
```

The `lifespan=lifespan` argument registers the startup/shutdown context manager.

## The `/predict` Endpoint

```python
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
```

### Step-by-Step Breakdown

**`customer: CustomerFeatures`** — FastAPI automatically parses the JSON request body into a `CustomerFeatures` object and validates it. If validation fails, the function is never called.

**`request: Request`** — FastAPI injects the raw request object, which gives access to `request.app.state.model`.

**`customer.model_dump()`** — Converts the Pydantic object to a plain Python dictionary: `{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}`.

**`pd.DataFrame([...], columns=FEATURE_COLUMNS)`** — Wraps the dictionary in a one-row DataFrame with columns in the exact order the model expects.

**`model.predict(features)[0]`** — Returns an array of predictions; `[0]` takes the first (and only) element.

**`model.classes_`** — A scikit-learn attribute listing the class labels in the order the model uses internally. For a binary classifier trained on `{0, 1}`, this is `[0, 1]`. The index of `1` in this list is the column index for the churn probability in `predict_proba`.

**`model.predict_proba(features)[0][positive_class_index]`** — `predict_proba` returns a 2D array where each row is a sample and each column is a class probability. `[0]` selects the first sample; `[positive_class_index]` selects the probability for class `1` (churn).

## Interactive API Documentation

FastAPI automatically generates interactive documentation at:

- `http://localhost:8000/docs` — Swagger UI (try requests in the browser)
- `http://localhost:8000/redoc` — ReDoc (read-only reference)

## Running the API

```bash
# With a .env file
uvicorn --env-file .env src.api:app --host 0.0.0.0 --port 8000

# Example request
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}'

# Example response
{"churn_probability": 0.63, "churn_label": 1}
```

---

Previous: [Model Evaluation ←](08-model-evaluation.md) | Next: [Remote Prediction →](10-remote-prediction.md)
