from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
FEATURE_COLUMNS = ["account_age", "monthly_spend", "support_tickets"]
TARGET_COLUMN = "churn"


def load_config():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def generate_data(n_rows=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    account_age = rng.integers(1, 73, size=n_rows)
    monthly_spend = np.round(np.clip(rng.normal(100, 30, size=n_rows), 20, 250), 2)
    support_tickets = rng.poisson(2.5, size=n_rows)

    churn_logit = (
        -1.3
        - 0.025 * (account_age - 36)
        + 0.012 * (monthly_spend - 100)
        + 0.45 * (support_tickets - 2.5)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "account_age": account_age,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            TARGET_COLUMN: churn,
        }
    )


def process_data():
    config = load_config()
    data_config = config["data"]
    raw_path = PROJECT_ROOT / data_config["raw_path"]
    train_path = PROJECT_ROOT / data_config["train_path"]
    test_path = PROJECT_ROOT / data_config["test_path"]

    dataset = generate_data()
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(raw_path, index=False)

    train_data, test_data = train_test_split(
        dataset,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        stratify=dataset[TARGET_COLUMN],
    )
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Saved {len(train_data)} training and {len(test_data)} test rows.")


if __name__ == "__main__":
    process_data()