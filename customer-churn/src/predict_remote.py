from dotenv import load_dotenv

load_dotenv()

import os

import mlflow
import mlflow.pyfunc
import pandas as pd


DATABRICKS_EXPERIMENT_PATH = os.getenv(
    "DATABRICKS_EXPERIMENT_PATH", "customer-churn"
)
MODEL_ARTIFACT_NAME = "random_forest"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]


def predict_latest_run():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("MLFLOW_TRACKING_URI must be set in the environment or .env.")
    if not os.environ.get("DATABRICKS_TOKEN"):
        raise RuntimeError("DATABRICKS_TOKEN must be set in the environment or .env.")

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(DATABRICKS_EXPERIMENT_PATH)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{DATABRICKS_EXPERIMENT_PATH}' was not found."
        )

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No completed runs found for '{DATABRICKS_EXPERIMENT_PATH}'."
        )

    run_id = runs.iloc[0]["run_id"]
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    model = mlflow.pyfunc.load_model(model_uri)

    customer = pd.DataFrame(
        [
            {
                "account_age": 12,
                "monthly_spend": 89.5,
                "support_tickets": 3,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
    prediction = model.predict(customer)
    print(f"Churn prediction for run {run_id}: {prediction[0]}")
    return prediction[0]


if __name__ == "__main__":
    predict_latest_run()