"""
Data generation and preprocessing pipeline for the customer-churn project.

Running this module as a script generates a synthetic dataset of SaaS
customer records, writes the raw CSV, and produces stratified train/test
splits ready for model training.

Usage
-----
    python src/data_processor.py
"""

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import FEATURE_COLUMNS, PROJECT_ROOT, TARGET_COLUMN
from src.utils import load_config

logger = logging.getLogger(__name__)


def generate_data(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """Generate a reproducible synthetic SaaS customer dataset.

    Each row represents one customer.  Feature distributions and the
    logistic churn model are designed to produce realistic class balance
    (~21 % churn rate) and meaningful feature-target relationships:

    * ``account_age``    — older accounts churn less (negative coefficient)
    * ``monthly_spend``  — higher spend correlates weakly with churn
    * ``support_tickets``— the strongest churn signal (positive coefficient)

    The churn label is sampled from a Bernoulli distribution whose
    probability is derived from the logistic model, introducing realistic
    label noise rather than a hard deterministic threshold.

    Parameters
    ----------
    n_rows:
        Number of customer records to generate.  Defaults to 1 000.
    random_state:
        Seed for the NumPy random number generator.  A fixed seed
        guarantees the same dataset on every run.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``account_age``, ``monthly_spend``,
        ``support_tickets``, and ``churn``.  No null values are present.
    """
    rng = np.random.default_rng(random_state)

    # --- Feature generation ------------------------------------------------
    # account_age: uniform integers 1–72 months (1 month to 6 years).
    account_age = rng.integers(1, 73, size=n_rows)

    # monthly_spend: normal around $100 with std $30, clipped to a
    # realistic $20–$250 range and rounded to cents.
    monthly_spend = np.round(
        np.clip(rng.normal(100, 30, size=n_rows), 20, 250), 2
    )

    # support_tickets: Poisson count data with mean 2.5 tickets.
    support_tickets = rng.poisson(2.5, size=n_rows)

    # --- Churn label generation --------------------------------------------
    # Logit is a linear combination of mean-centred features.
    # Intercept -1.3 → baseline churn probability ≈ 21 % at feature means.
    churn_logit = (
        -1.3
        - 0.025 * (account_age - 36)
        + 0.012 * (monthly_spend - 100)
        + 0.45 * (support_tickets - 2.5)
    )
    churn_probability = 1 / (1 + np.exp(-churn_logit))

    # Bernoulli draw: adds realistic noise so not every high-risk customer
    # churns and not every low-risk customer stays.
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "account_age": account_age,
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            TARGET_COLUMN: churn,
        }
    )


def process_data() -> None:
    """Generate data, write the raw CSV, and produce train/test splits.

    Reads file paths and split parameters from ``configs/config.yaml``.
    Output directories are created automatically if they do not exist.
    The split is stratified on the churn label to preserve class balance
    in both partitions.

    Raises
    ------
    FileNotFoundError
        If ``configs/config.yaml`` cannot be found.
    """
    config = load_config()
    data_config = config["data"]

    raw_path = PROJECT_ROOT / data_config["raw_path"]
    train_path = PROJECT_ROOT / data_config["train_path"]
    test_path = PROJECT_ROOT / data_config["test_path"]

    dataset = generate_data()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(raw_path, index=False)
    logger.info("Wrote raw dataset (%d rows) to %s", len(dataset), raw_path)

    train_data, test_data = train_test_split(
        dataset,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
        # Stratify preserves the churn ratio in both splits, which is
        # important when the positive class is a minority (~21 %).
        stratify=dataset[TARGET_COLUMN],
    )
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info(
        "Saved %d training rows to %s", len(train_data), train_path
    )
    logger.info("Saved %d test rows to %s", len(test_data), test_path)
    print(f"Saved {len(train_data)} training and {len(test_data)} test rows.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    process_data()
