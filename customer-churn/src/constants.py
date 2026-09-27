"""
Shared constants for the customer-churn pipeline.

Centralising these values ensures that every module — data processing,
training, evaluation, serving, and remote prediction — refers to the
same column names, paths, and artifact keys without copy-paste drift.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------

# Resolved at import time so every module gets the same absolute root
# regardless of the working directory from which it is invoked.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "config.yaml"

# ---------------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------------

# The three input features expected by the model.  Order matters: the
# DataFrame passed to sklearn must present columns in this exact sequence.
FEATURE_COLUMNS: list[str] = ["account_age", "monthly_spend", "support_tickets"]

# Binary target: 1 = churned, 0 = retained.
TARGET_COLUMN: str = "churn"

# ---------------------------------------------------------------------------
# MLflow artifact keys
# ---------------------------------------------------------------------------

# The artifact sub-path used when logging and loading the trained model.
# Both train.py and the serving/prediction scripts must use the same value.
MODEL_ARTIFACT_NAME: str = "random_forest"
