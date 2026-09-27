# Code Review Report — Customer Churn MLOps Blueprint

**Reviewer:** Senior Software Engineer / Technical Lead  
**Date:** 2025  
**Branch:** `main`  
**Scope:** `customer-churn/` blueprint — all source, test, configuration, and infrastructure files

---

## 1. Project Overview and Purpose

The `customer-churn` blueprint is a self-contained MLOps template that demonstrates an end-to-end machine learning workflow for a SaaS churn prediction problem. It generates synthetic customer data, trains a Random Forest classifier, tracks experiments with MLflow, serves predictions via a FastAPI REST API, versions data with DVC, packages the service in Docker, and automates testing and publishing through GitHub Actions.

The project is correctly scoped as a **portfolio blueprint**, not a production system. The review evaluates it against standards appropriate for that purpose: clean, readable, well-structured code that a developer can learn from, extend, and deploy with confidence.

---

## 2. Architecture and Folder Structure

### Before Refactoring

```
customer-churn/
├── configs/config.yaml
├── src/
│   ├── api.py
│   ├── data_processor.py
│   ├── evaluate.py
│   ├── predict_remote.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   └── test_data.py
├── .env.example
├── Dockerfile
├── requirements.txt
└── setup_data_tracking.sh
```

### After Refactoring

```
customer-churn/
├── configs/config.yaml
├── src/
│   ├── __init__.py          ← NEW: explicit package marker
│   ├── constants.py         ← NEW: single source of truth for shared constants
│   ├── utils.py             ← NEW: shared config loading and MLflow run discovery
│   ├── api.py               ← UPDATED
│   ├── data_processor.py    ← UPDATED
│   ├── evaluate.py          ← UPDATED (also fixed a functional bug)
│   ├── predict_remote.py    ← UPDATED
│   └── train.py             ← UPDATED
├── tests/
│   ├── __init__.py          ← NEW: explicit package marker
│   ├── test_api.py          ← UPDATED (expanded coverage)
│   └── test_data.py         ← UPDATED (expanded coverage)
├── .env.example
├── Dockerfile
├── pytest.ini               ← NEW: explicit test configuration
├── requirements.txt         ← UPDATED
└── setup_data_tracking.sh
```

The overall architecture (five-layer pipeline: data → training → serving → versioning → CI/CD) is sound and well-suited to the project's scope. No structural reorganisation was needed beyond adding the two new shared modules.

---

## 3. Code Quality Assessment

### 3.1 Duplicated Constants — FIXED

**Affected files:** `data_processor.py`, `train.py`, `evaluate.py`, `api.py`, `predict_remote.py`

**Problem:** Five constants were copy-pasted across all five source files:

```python
# Repeated verbatim in every file:
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"
MODEL_ARTIFACT_NAME = "random_forest"  # in train.py, api.py, predict_remote.py
```

**Why it matters:** Any change to a column name, path, or artifact key requires editing five files. A single typo in one file causes a silent mismatch that is hard to diagnose.

**Fix:** All five constants moved to `src/constants.py`. Every module now imports from there.

---

### 3.2 Duplicated Config-Loading Logic — FIXED

**Affected files:** `data_processor.py`, `train.py`, `evaluate.py`

**Problem:** `data_processor.py` defined a `load_config()` function. `train.py` and `evaluate.py` inlined the same `yaml.safe_load` pattern rather than importing the existing function.

**Fix:** `load_config()` moved to `src/utils.py` and imported by all three files.

---

### 3.3 Duplicated MLflow Run-Discovery Logic — FIXED

**Affected files:** `api.py`, `predict_remote.py`

**Problem:** Both files contained an identical 10-line block to find the latest finished MLflow run:

```python
experiment = mlflow.get_experiment_by_name(...)
if experiment is None:
    raise RuntimeError(...)
runs = mlflow.search_runs(...)
if runs.empty:
    raise RuntimeError(...)
run_id = runs.iloc[0]["run_id"]
```

**Why it matters:** The error messages and filter logic were slightly inconsistent between the two copies, and any future change (e.g., adding a tag filter) would need to be applied in two places.

**Fix:** Extracted to `get_latest_run_id(experiment_name)` in `src/utils.py`. Both `api.py` and `predict_remote.py` now call this single function.

---

### 3.4 Import Order Violation — FIXED

**Affected files:** `train.py`, `predict_remote.py`

**Problem:** Both files called `load_dotenv()` at module level before `import os` and other stdlib imports, violating PEP 8 import ordering (stdlib before third-party). While the early call is functionally necessary, the surrounding imports were not organised correctly.

**Fix:** `load_dotenv()` call is preserved at the top (it must precede any library that reads `os.environ` at import time), but all subsequent imports are now ordered: stdlib → third-party → local, with a comment explaining why `load_dotenv()` precedes them.

---

### 3.5 `EXPERIMENT_NAME` Hardcoded in `api.py` — FIXED

**Affected file:** `api.py`

**Problem:** `api.py` hardcoded `EXPERIMENT_NAME = "customer-churn"` as a module-level string literal, while `train.py` and `predict_remote.py` correctly read the same value from `DATABRICKS_EXPERIMENT_PATH` in the environment. This meant the API would always connect to the `"customer-churn"` experiment regardless of the environment variable, breaking Databricks deployments where the path is `/Users/<email>/customer-churn`.

**Fix:** `api.py` now reads `EXPERIMENT_NAME` from `os.environ.get("DATABRICKS_EXPERIMENT_PATH", "customer-churn")`, consistent with the other scripts.

---

### 3.6 No Logging — FIXED

**Affected files:** All five source files

**Problem:** All scripts used bare `print()` for output. In production, `print()` cannot be controlled by log level, cannot be redirected to a log aggregator, and is not captured by standard test frameworks.

**Fix:** All files now use `logging.getLogger(__name__)`. Informational messages (data paths, MLflow URIs, run IDs, metrics) are logged at `INFO`. Debug-level details use `DEBUG`. The `print()` calls that are part of the documented CLI output (e.g., `"Saved 800 training and 200 test rows."`) are retained alongside their `logger.info` equivalents so the user-facing behaviour is unchanged.

---

### 3.7 No Docstrings — FIXED

**Affected files:** All five source files

**Problem:** No module, function, or class had a docstring. A developer reading the code for the first time had no structured description of what each function does, what it expects, or what it raises.

**Fix:** Module-level docstrings added to all five files (describing purpose, usage, and environment variables). Function-level docstrings added to all public functions following NumPy/Google style with Parameters, Returns, and Raises sections. Inline comments added for non-obvious implementation decisions (e.g., why `stratify=` is used, why `skops_trusted_types` is needed, why `pyfunc` vs `sklearn` flavour is chosen).

---

## 4. Architecture and Design Assessment

### 4.1 Separation of Concerns

The five-script pipeline (generate → train → evaluate → serve → verify) has clean separation. Each script has a single responsibility and communicates with others through files or the MLflow API, not direct function calls. This is appropriate for the project's size.

### 4.2 Shared Module Design

The new `constants.py` and `utils.py` modules follow the principle of a single source of truth without introducing unnecessary abstraction. They contain only what is genuinely shared — no speculative generality.

### 4.3 `evaluate.py` — Functional Bug Fixed

**Problem:** `evaluate.py` loaded a model from `models/random_forest.joblib`, a path that `train.py` never writes to. The script would always fail with `FileNotFoundError` unless the user manually exported the model with `joblib.dump()`. This was a broken workflow with no documentation explaining the manual step.

**Fix:** `evaluate.py` now loads the model from MLflow using `get_latest_run_id()` and `mlflow.sklearn.load_model()`, consistent with how `api.py` and `predict_remote.py` work. The `joblib` import is removed. The script now works out-of-the-box after a training run.

---

## 5. Documentation Assessment

The `docs/` folder (added in a prior commit) is comprehensive and accurate. Module-level and function-level docstrings added during this review complement the external documentation by providing inline context for developers reading the source code directly.

No changes to `README.md` or `docs/` were required — they accurately describe the project.

---

## 6. Testing and Reliability Assessment

### 6.1 Missing `httpx` Dependency — FIXED

**Problem:** `fastapi.testclient.TestClient` requires `httpx` as its HTTP transport backend. It was not listed in `requirements.txt`, causing `ImportError` in environments where `httpx` was not already installed as a transitive dependency.

**Fix:** `httpx` added to `requirements.txt`.

### 6.2 No `pytest.ini` — FIXED

**Problem:** Test discovery, `pythonpath`, and output format relied entirely on pytest defaults. Running `pytest` from the wrong directory or without the correct `sys.path` would fail silently or with confusing import errors.

**Fix:** `pytest.ini` added with explicit `testpaths = tests`, `pythonpath = .`, and `addopts = -q --tb=short`.

### 6.3 Thin Test Coverage — EXPANDED

**`test_data.py` before:** One test with three assertions.

**`test_data.py` after:** Nine focused tests covering:
- Row count
- Column names and order
- No null values
- `account_age` range (1–72)
- `monthly_spend` range (20–250)
- `support_tickets` non-negative
- Churn label is binary (0 or 1 only)
- Reproducibility (same seed → same data)
- Different seeds produce different data

**`test_api.py` before:** One happy-path test.

**`test_api.py` after:** Seven tests covering:
- Happy path (HTTP 200, correct JSON body)
- Five invalid-input cases (HTTP 422): `account_age=0`, negative age, negative spend, negative tickets, missing required field
- Startup failure when experiment not found
- Startup failure when no finished runs exist

### 6.4 Remaining Coverage Gaps

The following are not tested and would require additional work:

| Component | Gap | Reason Not Fixed Here |
|-----------|-----|-----------------------|
| `train.py` | Full training loop | Requires MLflow server or extensive mocking |
| `evaluate.py` | Full evaluation | Requires MLflow server |
| `predict_remote.py` | Remote prediction | Requires live Databricks credentials |
| `data_processor.process_data()` | File I/O path | Requires temp directory fixture |

---

## 7. Security and Configuration Assessment

### 7.1 Secret Handling — Already Correct

`.env` is in `.gitignore`. `.env.example` contains only placeholder values. The Dockerfile excludes `.env` via `.dockerignore`. No credentials appear in source code. This is correct.

### 7.2 `skops_trusted_types` — Already Correct

The explicit `skops_trusted_types=["sklearn.tree._tree.Tree"]` in `train.py` is the correct security practice for MLflow sklearn model serialisation. A comment was added explaining why this is necessary.

### 7.3 Unpinned Dependencies

`requirements.txt` has no version pins. For a portfolio project this is acceptable — it keeps installation simple and avoids stale pins. For a production deployment, a `pip freeze > requirements-lock.txt` or a tool like `pip-compile` would be appropriate. This is documented as a known limitation, not fixed.

### 7.4 `uvicorn[standard]` — FIXED

`requirements.txt` listed `uvicorn` without the `[standard]` extra. The `[standard]` extra includes `uvloop` and `httptools` for better performance, and is the recommended installation for production use.

---

## 8. Maintainability and Scalability

### What Scales Well

- The `constants.py` / `utils.py` pattern means adding a new script to the pipeline requires zero duplication of shared values.
- The `get_latest_run_id()` utility can be extended (e.g., to filter by tag or metric threshold) in one place.
- `pytest.ini` makes the test suite runnable from any directory without path manipulation.

### What Would Need Attention for a Real Production System

- **No model registry:** The API always loads the most recent finished run. A real system would use the MLflow Model Registry with explicit staging (`Staging` → `Production`) to control which model is served.
- **No API authentication:** The `/predict` endpoint has no authentication. Any caller can make predictions.
- **No request-level logging:** The API logs at startup but not per-request. A production API would log request IDs, latency, and prediction outcomes.
- **Single-worker serving:** Uvicorn is started with default settings (one worker). A production deployment would use `--workers N` or a process manager like Gunicorn.

---

## 9. Identified Bugs and Technical Debt

| # | File | Issue | Severity | Fixed |
|---|------|-------|----------|-------|
| 1 | `evaluate.py` | Loads from `models/random_forest.joblib` which is never written by the pipeline | **High** | ✅ Yes |
| 2 | `api.py` | `EXPERIMENT_NAME` hardcoded, ignores `DATABRICKS_EXPERIMENT_PATH` env var | **Medium** | ✅ Yes |
| 3 | All src files | Constants duplicated 4–5× across files | **Medium** | ✅ Yes |
| 4 | `api.py`, `predict_remote.py` | MLflow run-discovery logic duplicated | **Medium** | ✅ Yes |
| 5 | `requirements.txt` | `httpx` missing, tests fail in clean environments | **Medium** | ✅ Yes |
| 6 | All src files | No logging; bare `print()` only | **Low** | ✅ Yes |
| 7 | All src files | No docstrings | **Low** | ✅ Yes |
| 8 | `train.py`, `predict_remote.py` | Import order violation (PEP 8) | **Low** | ✅ Yes |
| 9 | `requirements.txt` | `uvicorn` missing `[standard]` extra | **Low** | ✅ Yes |
| 10 | No `pytest.ini` | Test discovery fragile without explicit config | **Low** | ✅ Yes |

---

## 10. Changes Implemented

| File | Change Type | Summary |
|------|-------------|---------|
| `src/constants.py` | **Created** | Centralised `PROJECT_ROOT`, `CONFIG_PATH`, `FEATURE_COLUMNS`, `TARGET_COLUMN`, `MODEL_ARTIFACT_NAME` |
| `src/utils.py` | **Created** | Shared `load_config()` and `get_latest_run_id()` with full docstrings |
| `src/__init__.py` | **Created** | Explicit package marker |
| `tests/__init__.py` | **Created** | Explicit package marker |
| `pytest.ini` | **Created** | Explicit `testpaths`, `pythonpath`, `addopts` |
| `src/data_processor.py` | **Refactored** | Uses `constants`, `utils`; added logging, docstrings, inline comments |
| `src/train.py` | **Refactored** | Uses `constants`, `utils`; fixed import order; added logging, docstrings |
| `src/evaluate.py` | **Bug fix + refactor** | Now loads from MLflow (not broken joblib path); uses `constants`, `utils`; added logging, docstrings |
| `src/api.py` | **Refactored** | Uses `constants`, `utils`; reads `EXPERIMENT_NAME` from env; added logging, docstrings, Field descriptions |
| `src/predict_remote.py` | **Refactored** | Uses `constants`, `utils`; fixed import order; added logging, docstrings |
| `tests/test_data.py` | **Expanded** | 1 test → 9 tests; added range, binary, reproducibility, and seed-difference checks |
| `tests/test_api.py` | **Expanded** | 1 test → 7 tests; added 5 invalid-input (422) cases and 2 startup-failure cases |
| `requirements.txt` | **Updated** | Added `httpx`; changed `uvicorn` → `uvicorn[standard]`; added section comments |

---

## 11. Test Execution Results

The test suite was run with `python -m pytest -v` from the `customer-churn/` directory after the OS-level AppLocker policy was resolved.

**Result: 17 passed, 0 warnings in 5.31s** ✅

```
platform win32 -- Python 3.14.4, pytest-9.1.1
configfile: pytest.ini  |  testpaths: tests
collected 17 items

tests/test_api.py  ........                                        [ 47%]
tests/test_data.py  .........                                      [100%]

17 passed in 5.31s
```

**Full pipeline verified end-to-end:**

| Step | Command | Result |
|------|---------|--------|
| Data generation | `python src/data_processor.py` | ✅ 800 train + 200 test rows written |
| Model training | `python src/train.py` | ✅ Run `59683cdee7fd4e188b341f5e54bcdacc` logged to Databricks |
| Evaluation | `python src/evaluate.py` | ✅ Classification report printed (accuracy 0.79) |
| Test suite | `python -m pytest -v` | ✅ 17/17 passed, 0 warnings |

---

## 12. Remaining Limitations and Future Improvements

### Known Limitations (Not Fixed — Out of Scope for This Review)

1. **No MLflow Model Registry integration.** The API loads the most recent finished run, which is fragile. A production system would promote models through `Staging` → `Production` stages.
2. **No API authentication.** The `/predict` endpoint is open to any caller.
3. **Unpinned dependencies.** Appropriate for a portfolio project; a production deployment needs pinned versions.
4. **`process_data()` not tested.** The file I/O path in `data_processor.py` is not covered by the test suite.
5. **Single Uvicorn worker.** The Dockerfile starts one worker. A production deployment needs `--workers` or Gunicorn.

### Suggested Future Improvements

- Add a `conftest.py` with a `tmp_path`-based fixture to test `process_data()` without writing to the real `data/` directory.
- Add a `GET /health` endpoint to the API for container health checks.
- Add `--workers` to the Dockerfile `CMD` or switch to a Gunicorn + Uvicorn worker setup.
- Consider `pip-compile` (from `pip-tools`) to generate a pinned `requirements.txt` from a `requirements.in` source file.
- Add per-request structured logging to the API (request ID, latency, prediction outcome).
