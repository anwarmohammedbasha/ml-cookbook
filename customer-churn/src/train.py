from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"
DATABRICKS_EXPERIMENT_PATH = os.getenv(
    "DATABRICKS_EXPERIMENT_PATH", "customer-churn"
)
MODEL_ARTIFACT_NAME = "random_forest"


def train():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    train_path = PROJECT_ROOT / config["data"]["train_path"]
    test_path = PROJECT_ROOT / config["data"]["test_path"]
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(DATABRICKS_EXPERIMENT_PATH)

    with mlflow.start_run() as run:
        model_parameters = {
            "n_estimators": config["model"]["n_estimators"],
            "max_depth": config["model"]["max_depth"],
            "random_state": config["model"]["random_state"],
        }
        mlflow.log_params(model_parameters)

        model = RandomForestClassifier(**model_parameters)
        model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])

        predictions = model.predict(test_data[FEATURE_COLUMNS])
        mlflow.log_metrics(
            {
                "accuracy": accuracy_score(test_data[TARGET_COLUMN], predictions),
                "precision": precision_score(
                    test_data[TARGET_COLUMN], predictions, zero_division=0
                ),
                "recall": recall_score(
                    test_data[TARGET_COLUMN], predictions, zero_division=0
                ),
            }
        )
        mlflow.sklearn.log_model(
            model,
            name=MODEL_ARTIFACT_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        print(f"Logged model to MLflow run {run.info.run_id}.")


if __name__ == "__main__":
    train()