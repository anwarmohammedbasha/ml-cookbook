# 01 — Project Overview

## What Is Customer Churn?

**Customer churn** is when a customer stops using a product or service. For a SaaS (Software as a Service) company — one that charges a recurring subscription fee — churn is one of the most important business metrics. Losing customers directly reduces revenue, and acquiring new customers is typically far more expensive than retaining existing ones.

A **churn prediction model** analyses historical customer behaviour and flags customers who are likely to cancel before they actually do. This gives the business time to intervene — for example, by offering a discount or reaching out with support.

## What This Project Does

The `customer-churn` blueprint builds a complete, production-ready machine learning system that:

1. **Generates** a synthetic dataset of 1,000 SaaS customer records.
2. **Trains** a binary classification model (Random Forest) to predict whether a customer will churn.
3. **Tracks** every training experiment — parameters, metrics, and the trained model artifact — using MLflow.
4. **Serves** predictions over an HTTP REST API built with FastAPI.
5. **Versions** the processed dataset using DVC so that data changes are reproducible.
6. **Packages** the entire service in a Docker container for consistent deployment.
7. **Automates** testing and container publishing through a GitHub Actions CI/CD pipeline.

## The Prediction Problem

This is a **binary classification** problem. Given three features about a customer, the model predicts one of two outcomes:

| Feature | Description |
|---------|-------------|
| `account_age` | How many months the customer has been subscribed (1–72) |
| `monthly_spend` | How much the customer pays per month in dollars (20–250) |
| `support_tickets` | How many support tickets the customer has raised |

The model outputs:

| Output | Description |
|--------|-------------|
| `churn_label` | `1` if the customer is predicted to churn, `0` if not |
| `churn_probability` | A probability between 0 and 1 indicating confidence in the churn prediction |

## Why Synthetic Data?

The project generates synthetic data rather than using a real dataset. This makes the blueprint:

- **Self-contained** — no external data download is required.
- **Reproducible** — the same random seed always produces the same dataset.
- **Safe** — no real customer PII (Personally Identifiable Information) is involved.

The synthetic data is generated using a logistic model that encodes realistic relationships: older accounts churn less, higher spend correlates slightly with churn, and more support tickets strongly predict churn.

## MLOps: What It Means and Why It Matters

**MLOps** (Machine Learning Operations) is the practice of applying software engineering and DevOps principles to machine learning systems. A model that only works in a notebook is not production-ready. MLOps addresses the full lifecycle:

| MLOps Concern | How This Project Addresses It |
|---------------|-------------------------------|
| Reproducibility | Fixed random seeds; DVC-tracked data; MLflow-logged parameters |
| Experiment tracking | MLflow logs every run's parameters, metrics, and model artifact |
| Model serving | FastAPI REST API loads the latest trained model at startup |
| Data versioning | DVC tracks processed CSVs as Git pointer files |
| Containerisation | Docker packages the app and its dependencies into a portable image |
| Automated testing | GitHub Actions runs pytest on every push and pull request |
| Continuous delivery | GitHub Actions builds and pushes the Docker image to GHCR on merges to `main` |

## Scope and Limitations

This is a **blueprint** — a learning template — not a production system for a real business. Specific limitations to be aware of:

- The data is synthetic. A real deployment would require real customer data and a proper data pipeline.
- The model is not registered in an MLflow Model Registry. The API always loads the most recent completed run, which is a simple but fragile strategy.
- There is no authentication on the API endpoints.
- `evaluate.py` loads a model from a local `.joblib` file, which is a separate path from the MLflow-based loading used by `api.py` and `predict_remote.py`.

---

Next: [System Architecture →](02-system-architecture.md)
