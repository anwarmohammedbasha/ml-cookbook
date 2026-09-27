# MLOps Cookbook — Documentation

Welcome to the technical documentation for the **MLOps Cookbook** repository. This guide walks you through every concept, component, and workflow in the project, from first principles to production deployment.

## What This Project Is

`ml-cookbook` is a portfolio of production-ready MLOps templates. Each blueprint demonstrates a complete, end-to-end machine learning workflow: data preparation, model training, experiment tracking, model serving, data versioning, and automated CI/CD delivery.

The first and currently complete blueprint is **Customer Churn**, which predicts whether a SaaS customer will cancel their subscription using a Scikit-Learn Random Forest classifier.

## Prerequisites

Before reading the documentation or running the project, you should be comfortable with:

- Python 3.11+
- Basic machine learning concepts (classification, train/test splits, metrics)
- Command-line usage
- Git basics

Familiarity with Docker, MLflow, and Databricks is helpful but not required — the documentation explains each tool from scratch.

---

## Documentation Roadmap

Read the documents in order for the best learning experience. Each document builds on the previous one.

### 1. Project Foundations

Start here to understand what the project does and how it is organised.

| # | Document | What You Will Learn |
|---|----------|---------------------|
| 01 | [Project Overview](01-project-overview.md) | What churn is, what the project does, and what MLOps means |
| 02 | [System Architecture](02-system-architecture.md) | How all components fit together, with Mermaid diagrams |
| 03 | [Project Structure](03-project-structure.md) | Every file and folder explained |
| 04 | [Technology Stack](04-technology-stack.md) | Every library and tool used, and why |

### 2. Configuration and Setup

| # | Document | What You Will Learn |
|---|----------|---------------------|
| 05 | [Configuration](05-configuration.md) | `config.yaml`, `.env`, environment variables, local vs. Databricks |

### 3. Implementation Walkthrough

Step-by-step code walkthroughs for every source file, in pipeline execution order.

| # | Document | What You Will Learn |
|---|----------|---------------------|
| 06 | [Data Pipeline](06-data-pipeline.md) | `data_processor.py`: synthetic data generation and train/test splitting |
| 07 | [Model Training](07-model-training.md) | `train.py`: Random Forest training and MLflow experiment logging |
| 08 | [Model Evaluation](08-model-evaluation.md) | `evaluate.py`: classification report and metric interpretation |
| 09 | [Prediction API](09-prediction-api.md) | `api.py`: FastAPI service, ASGI lifespan, and the `/predict` endpoint |
| 10 | [Remote Prediction](10-remote-prediction.md) | `predict_remote.py`: fetching a model from MLflow and predicting |

### 4. Operations

| # | Document | What You Will Learn |
|---|----------|---------------------|
| 11 | [Data Versioning with DVC](11-data-versioning.md) | What DVC is, pointer files, remotes, and `setup_data_tracking.sh` |
| 12 | [Testing](12-testing.md) | The test suite, `monkeypatch`, `TestClient`, and coverage gaps |
| 13 | [Docker and Containerisation](13-docker.md) | The Dockerfile line-by-line and the baked-in model pattern |
| 14 | [CI/CD with GitHub Actions](14-cicd.md) | The automated test and publish pipeline |
| — | [Code Review Report](code-review.md) | Full professional code review: findings, fixes, and test results |

### 5. Deep Dives — Algorithms, Mathematics, and Internals

These documents explain the theoretical and mathematical foundations of the key concepts used in the project. Read them after the implementation walkthrough for the deepest understanding.

| # | Document | What You Will Learn |
|---|----------|---------------------|
| A | [Random Forest: Theory and Implementation](A-random-forest-deep-dive.md) | Decision trees, Gini impurity, bootstrap sampling, bias-variance trade-off, feature importance |
| B | [Classification Metrics: Theory and Implementation](B-classification-metrics.md) | Confusion matrix, accuracy, precision, recall, F1-score, class imbalance |
| C | [Synthetic Data Generation: Mathematics and Design](C-synthetic-data-math.md) | Uniform, Normal, and Poisson distributions; the logistic/sigmoid function; Bernoulli sampling; stratified splitting |
| D | [MLflow Internals: Tracking, Artifacts, and Model Serialisation](D-mlflow-internals.md) | Tracking store schema, artifact store layout, model flavours, skops security, run lifecycle |

---

## Quick-Start Summary

```bash
# 1. Install dependencies
cd customer-churn
pip install -r requirements.txt

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Generate data
python src/data_processor.py

# 4. Train the model
python src/train.py

# 5. Serve the API
uvicorn --env-file .env src.api:app --host 0.0.0.0 --port 8000

# 6. Make a prediction
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"account_age": 12, "monthly_spend": 89.5, "support_tickets": 3}'
```

---

## Source Code Reference

| File | Purpose |
|------|---------|
| [`customer-churn/src/data_processor.py`](../customer-churn/src/data_processor.py) | Generates and splits data |
| [`customer-churn/src/train.py`](../customer-churn/src/train.py) | Trains the model and logs to MLflow |
| [`customer-churn/src/evaluate.py`](../customer-churn/src/evaluate.py) | Evaluates a locally saved model |
| [`customer-churn/src/api.py`](../customer-churn/src/api.py) | FastAPI prediction service |
| [`customer-churn/src/predict_remote.py`](../customer-churn/src/predict_remote.py) | Fetches model from MLflow and predicts |
| [`customer-churn/configs/config.yaml`](../customer-churn/configs/config.yaml) | Centralised hyperparameters and paths |
| [`customer-churn/Dockerfile`](../customer-churn/Dockerfile) | Container build instructions |
| [`customer-churn/setup_data_tracking.sh`](../customer-churn/setup_data_tracking.sh) | Initialises DVC tracking |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | GitHub Actions CI/CD pipeline |
