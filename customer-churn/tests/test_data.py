"""Tests for the synthetic data generation pipeline."""

import pandas as pd
import pytest

from src.data_processor import FEATURE_COLUMNS, TARGET_COLUMN, generate_data


@pytest.fixture(scope="module")
def dataset() -> pd.DataFrame:
    """Generate the default 1 000-row dataset once for all tests in this module."""
    return generate_data()


def test_row_count(dataset: pd.DataFrame) -> None:
    assert len(dataset) == 1000


def test_column_names_and_order(dataset: pd.DataFrame) -> None:
    assert list(dataset.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_no_null_values(dataset: pd.DataFrame) -> None:
    assert dataset.notna().all().all()


def test_account_age_range(dataset: pd.DataFrame) -> None:
    """account_age must be a positive integer between 1 and 72 inclusive."""
    assert dataset["account_age"].between(1, 72).all()


def test_monthly_spend_range(dataset: pd.DataFrame) -> None:
    """monthly_spend must be clipped to the [20, 250] range."""
    assert dataset["monthly_spend"].between(20, 250).all()


def test_support_tickets_non_negative(dataset: pd.DataFrame) -> None:
    """support_tickets is a Poisson count and must be >= 0."""
    assert (dataset["support_tickets"] >= 0).all()


def test_churn_is_binary(dataset: pd.DataFrame) -> None:
    """Churn label must be 0 or 1 only."""
    assert set(dataset[TARGET_COLUMN].unique()).issubset({0, 1})


def test_reproducibility() -> None:
    """Two calls with the same random_state must produce identical datasets."""
    df1 = generate_data(random_state=0)
    df2 = generate_data(random_state=0)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_differ() -> None:
    """Different random states must produce different datasets."""
    df1 = generate_data(random_state=1)
    df2 = generate_data(random_state=2)
    assert not df1.equals(df2)
