from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"


def train():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    train_path = PROJECT_ROOT / config["data"]["train_path"]
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
    train_data = pd.read_csv(train_path)

    model = RandomForestClassifier(
        n_estimators=config["model"]["n_estimators"],
        max_depth=config["model"]["max_depth"],
        random_state=config["model"]["random_state"],
    )
    model.fit(train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN])

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved trained model to {model_path.relative_to(PROJECT_ROOT)}.")


if __name__ == "__main__":
    train()