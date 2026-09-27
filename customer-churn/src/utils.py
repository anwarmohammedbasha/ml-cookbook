"""
Shared utility functions used across the customer-churn pipeline.

Keeping these helpers in one place avoids copy-paste duplication and
ensures that changes to config loading or MLflow run-discovery propagate
to every consumer automatically.
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import pandas as pd
import yaml

from src.constants import CONFIG_PATH

logger = logging.getLogger(__name__)


def load_config() -> dict[str, Any]:
    """Load and return the parsed YAML configuration file.

    Returns
    -------
    dict[str, Any]
        The full configuration dictionary with ``data``, ``model``, and
        ``split`` sections.

    Raises
    ------
    FileNotFoundError
        If ``configs/config.yaml`` does not exist at the expected path.
    yaml.YAMLError
        If the file exists but cannot be parsed as valid YAML.
    """
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def get_latest_run_id(experiment_name: str) -> str:
    """Return the run ID of the most recent finished MLflow run.

    Searches the given experiment for runs with status ``FINISHED``,
    ordered by start time descending, and returns the ID of the first
    result.  Both the API startup and the remote-prediction script use
    this same discovery logic.

    Parameters
    ----------
    experiment_name:
        The MLflow experiment name (or Databricks workspace path) to
        search within.

    Returns
    -------
    str
        The run ID string, e.g. ``"a1b2c3d4e5f6..."``.

    Raises
    ------
    RuntimeError
        If the experiment does not exist or has no finished runs.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{experiment_name}' was not found. "
            "Run train.py at least once before serving or predicting."
        )

    runs: pd.DataFrame = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError(
            f"No completed runs found for experiment '{experiment_name}'. "
            "Ensure train.py has completed successfully."
        )

    run_id: str = runs.iloc[0]["run_id"]
    logger.debug("Resolved latest finished run: %s", run_id)
    return run_id
