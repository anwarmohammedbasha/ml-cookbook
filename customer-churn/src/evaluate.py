from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.metrics import classification_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"


def evaluate():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    test_path = PROJECT_ROOT / config["data"]["test_path"]
    model_path = PROJECT_ROOT / "models" / "random_forest.joblib"
    test_data = pd.read_csv(test_path)
    model = joblib.load(model_path)
    predictions = model.predict(test_data[FEATURE_COLUMNS])

    print(classification_report(test_data[TARGET_COLUMN], predictions))


if __name__ == "__main__":
    evaluate()