# MLOps Cookbook

`ml-cookbook` is a portfolio of production-ready MLOps templates. Each blueprint demonstrates an end-to-end path from data preparation and model training through evaluation, serving, versioning, and automated delivery.

## Projects

### 1. Customer Churn (Completed)

The [`customer-churn`](customer-churn/README.md) blueprint predicts SaaS customer churn from account age, monthly spend, and support-ticket activity. It combines a Scikit-Learn Random Forest, MLflow experiment tracking on Databricks, a FastAPI prediction service, and Docker packaging.

- **CI/CD:** GitHub Actions runs pytest on pushes and pull requests to `main`; successful pushes to `main` build and publish the image to GitHub Container Registry (GHCR).
- **Data versioning:** DVC tracks processed datasets through Git pointer files and stores dataset contents in its cache or a configured DVC remote.
- **Experiment tracking:** MLflow logs model parameters, evaluation metrics, and model artifacts to the configured Databricks experiment.

See the project guide for setup, execution, and deployment instructions.