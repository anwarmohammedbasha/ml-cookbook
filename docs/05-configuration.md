# 05 — Configuration

The project uses two separate configuration mechanisms: a YAML file for static pipeline settings and environment variables for secrets and deployment-specific values.

## `configs/config.yaml` — Static Pipeline Settings

This file is the single source of truth for all tunable parameters. It is read by `data_processor.py`, `train.py`, and `evaluate.py` using `yaml.safe_load()`.

```yaml
data:
  raw_path: data/raw/customers.csv
  train_path: data/processed/train.csv
  test_path: data/processed/test.csv

model:
  n_estimators: 200
  max_depth: 8
  random_state: 42

split:
  test_size: 0.2
  random_state: 42
```

### `data` Section

| Key | Value | Meaning |
|-----|-------|---------|
| `raw_path` | `data/raw/customers.csv` | Where the full 1,000-row dataset is written |
| `train_path` | `data/processed/train.csv` | Where the 800-row training split is written |
| `test_path` | `data/processed/test.csv` | Where the 200-row test split is written |

All paths are relative to the `customer-churn/` project root. The code resolves them to absolute paths using `Path(__file__).resolve().parents[1]`.

### `model` Section

| Key | Value | Meaning |
|-----|-------|---------|
| `n_estimators` | `200` | Number of decision trees in the Random Forest |
| `max_depth` | `8` | Maximum depth of each tree (limits overfitting) |
| `random_state` | `42` | Seed for reproducible tree building |

These values are passed directly to `RandomForestClassifier(**model_parameters)` in `train.py` and are also logged to MLflow with `mlflow.log_params()`.

### `split` Section

| Key | Value | Meaning |
|-----|-------|---------|
| `test_size` | `0.2` | 20% of the data is held out for testing (200 rows) |
| `random_state` | `42` | Seed for reproducible splitting |

The split uses stratification on the `churn` column, which ensures the class ratio is preserved in both splits.

### How Config Is Loaded

Every script that needs configuration calls the same pattern:

```python
# From data_processor.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
```

`Path(__file__).resolve().parents[1]` navigates two levels up from the current file (`src/data_processor.py` → `src/` → `customer-churn/`), giving the project root regardless of where the script is invoked from.

---

## Environment Variables — Secrets and Deployment Settings

Environment variables hold values that differ between environments (local, CI, production) or that must never be committed to source control (tokens, passwords).

### `.env.example` — The Template

```dotenv
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://<your-databricks-workspace>
DATABRICKS_TOKEN=<your-databricks-token>
DATABRICKS_EXPERIMENT_PATH=customer-churn
```

Copy this to `.env` and fill in real values. The `.env` file is listed in `.gitignore` and will never be committed.

### Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MLFLOW_TRACKING_URI` | Yes | `sqlite:///mlflow.db` | Where MLflow stores run data. Set to `databricks` for Databricks, or a `sqlite:///` path for local use |
| `DATABRICKS_HOST` | If using Databricks | — | The base URL of your Databricks workspace, e.g. `https://community.cloud.databricks.com` |
| `DATABRICKS_TOKEN` | If using Databricks | — | A personal access token from your Databricks workspace |
| `DATABRICKS_EXPERIMENT_PATH` | No | `customer-churn` | The MLflow experiment name or path. On Databricks, use a full path like `/Users/<email>/customer-churn` |
| `MLFLOW_TRACKING_TOKEN` | No | — | A bearer token for self-hosted MLflow servers. Not used by the Databricks SDK path |

### How Variables Are Loaded

`train.py` and `predict_remote.py` call `load_dotenv()` at the very top of the file, before any other imports that might read environment variables:

```python
# From train.py
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ before anything else

import os
# ...
tracking_uri = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
```

The `os.environ.get(key, default)` pattern means the code works without a `.env` file by falling back to a local SQLite database.

`api.py` does not call `load_dotenv()` because it is launched with `uvicorn --env-file .env`, which loads the file before the application starts.

### Local vs. Databricks Tracking

The project supports two MLflow backends:

**Local SQLite (default, no account needed):**
```dotenv
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```
MLflow creates `mlflow.db` in the `customer-churn/` directory. The experiment name is just `customer-churn`.

**Databricks (remote, requires account):**
```dotenv
MLFLOW_TRACKING_URI=databricks
DATABRICKS_HOST=https://community.cloud.databricks.com
DATABRICKS_TOKEN=dapi...
DATABRICKS_EXPERIMENT_PATH=/Users/you@example.com/customer-churn
```
MLflow uses the Databricks SDK to authenticate and store runs in the Databricks workspace.

---

Previous: [Technology Stack ←](04-technology-stack.md) | Next: [Data Pipeline →](06-data-pipeline.md)
