# 07 — Model Training

**Source file:** [`customer-churn/src/train.py`](../customer-churn/src/train.py)

## Responsibilities

`train.py` reads the processed training and test CSVs, trains a Random Forest classifier, evaluates it on the test set, and logs the entire run — parameters, metrics, and the model artifact — to MLflow.

## Prerequisites

`data_processor.py` must have run first to create `data/processed/train.csv` and `data/processed/test.csv`.

## Module-Level Setup

```python
from dotenv import load_dotenv
load_dotenv()
```

`load_dotenv()` is called before any other imports. This ensures that `MLFLOW_TRACKING_URI`, `DATABRICKS_HOST`, and `DATABRICKS_TOKEN` are in `os.environ` before MLflow initialises.

```python
DATABRICKS_EXPERIMENT_PATH = os.getenv("DATABRICKS_EXPERIMENT_PATH", "customer-churn")
MODEL_ARTIFACT_NAME = "random_forest"
```

`DATABRICKS_EXPERIMENT_PATH` defaults to `"customer-churn"` if the environment variable is not set, which works for local SQLite tracking. `MODEL_ARTIFACT_NAME` is the key under which the model is stored in the MLflow run — `api.py` and `predict_remote.py` use the same constant to load it.

## The `train()` Function

### Step 1 — Load Config and Data

```python
with CONFIG_PATH.open(encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

train_data = pd.read_csv(PROJECT_ROOT / config["data"]["train_path"])
test_data = pd.read_csv(PROJECT_ROOT / config["data"]["test_path"])
```

### Step 2 — Configure MLflow

```python
tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(DATABRICKS_EXPERIMENT_PATH)
```

`mlflow.set_tracking_uri()` tells MLflow where to store run data. `mlflow.set_experiment()` creates the experiment if it does not exist, or selects it if it does. All subsequent `mlflow.*` calls within this process will write to this experiment.

### Step 3 — Start an MLflow Run

```python
with mlflow.start_run() as run:
```

`mlflow.start_run()` is a context manager. Everything inside the `with` block is associated with a single run. The run is automatically marked as `FINISHED` when the block exits normally, or `FAILED` if an exception is raised.

### Step 4 — Log Hyperparameters

```python
model_parameters = {
    "n_estimators": config["model"]["n_estimators"],
    "max_depth": config["model"]["max_depth"],
    "random_state": config["model"]["random_state"],
}
mlflow.log_params(model_parameters)
```

`mlflow.log_params()` records the hyperparameters used for this run. This is what makes runs comparable — you can look at the MLflow UI and see exactly which parameters produced which metrics.

### Step 5 — Train the Model

```python
model = RandomForestClassifier(**model_parameters)
model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])
```

`**model_parameters` unpacks the dictionary as keyword arguments, equivalent to writing `RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)`.

`model.fit(X, y)` trains the Random Forest on the feature matrix `X` (the three feature columns) and the target vector `y` (the `churn` column).

### Step 6 — Evaluate and Log Metrics

```python
predictions = model.predict(test_data[FEATURE_COLUMNS])
mlflow.log_metrics({
    "accuracy": accuracy_score(test_data[TARGET_COLUMN], predictions),
    "precision": precision_score(test_data[TARGET_COLUMN], predictions, zero_division=0),
    "recall": recall_score(test_data[TARGET_COLUMN], predictions, zero_division=0),
})
```

The model predicts on the **test set** (data it has never seen during training). Three metrics are computed and logged:

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Percentage of all predictions that are correct |
| **Precision** | Of all customers predicted to churn, what fraction actually churned |
| **Recall** | Of all customers who actually churned, what fraction were correctly predicted |

`zero_division=0` prevents a warning if the model predicts no positive cases (which can happen with very small datasets or extreme class imbalance).

### Step 7 — Log the Model Artifact

```python
mlflow.sklearn.log_model(
    model,
    name=MODEL_ARTIFACT_NAME,
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)
```

`mlflow.sklearn.log_model()` serialises the trained model and uploads it to the MLflow tracking store under the key `"random_forest"`. The `skops_trusted_types` argument is a security parameter that explicitly allows the internal scikit-learn tree type to be deserialised when the model is loaded later.

## What Is a Random Forest?

A **Random Forest** is an ensemble of decision trees. Each tree is trained on a random subset of the training data (bootstrap sampling) and considers only a random subset of features at each split. The final prediction is the majority vote across all trees.

Key advantages for this use case:
- Handles the mix of integer and float features without scaling
- Robust to outliers in `monthly_spend`
- Provides `predict_proba()` for probability estimates, not just binary labels
- The `random_state` parameter makes training reproducible

For a complete explanation of the algorithm — including decision tree splitting, Gini impurity, bootstrap sampling, the bias-variance trade-off, and feature importance — see the [Random Forest Deep Dive →](A-random-forest-deep-dive.md).

## MLflow Concepts

### Experiment

An **experiment** is a named collection of runs. All runs from `train.py` go into the experiment named by `DATABRICKS_EXPERIMENT_PATH`. You can compare runs within an experiment in the MLflow UI.

### Run

A **run** is a single execution of `train()`. It records:
- **Parameters** — the hyperparameters used
- **Metrics** — the evaluation scores
- **Artifacts** — the serialised model file

### Artifact URI

The model is stored at `runs:/<run_id>/random_forest`. This URI is used by `api.py` and `predict_remote.py` to load the model:

```python
mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}")
```

For a complete explanation of how MLflow stores data internally — the tracking store schema, artifact store layout, model flavours, skops serialisation, and the run lifecycle state machine — see [MLflow Internals →](D-mlflow-internals.md).

## Execution

```bash
cd customer-churn
python src/train.py
# Output: Logged model to MLflow run <run_id>.
```

---

Previous: [Data Pipeline ←](06-data-pipeline.md) | Next: [Model Evaluation →](08-model-evaluation.md)
