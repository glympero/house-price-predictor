"""Tests for dataset loading and schema validation."""

import pandas as pd
import pytest

from house_prices import config
from house_prices.data import load_dataset, validate_dataset


@pytest.fixture
def valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_date": [2012.917, 2013.5],
            "house_age_years": [32.0, 5.1],
            "mrt_distance_m": [84.878, 2175.03],
            "n_convenience_stores": [10, 3],
            "latitude": [24.983, 24.963],
            "longitude": [121.540, 121.512],
            "price_per_unit_area": [37.9, 21.4],
        }
    )


def test_validate_accepts_valid_frame(valid_frame):
    validate_dataset(valid_frame)


def test_validate_rejects_missing_column(valid_frame):
    with pytest.raises(ValueError, match="columns are"):
        validate_dataset(valid_frame.drop(columns=["latitude"]))


def test_validate_rejects_nulls(valid_frame):
    valid_frame.loc[0, "house_age_years"] = None
    with pytest.raises(ValueError, match="null values"):
        validate_dataset(valid_frame)


def test_validate_rejects_out_of_range(valid_frame):
    valid_frame.loc[0, "latitude"] = 90.0  # far outside the Taipei region
    with pytest.raises(ValueError, match="outside"):
        validate_dataset(valid_frame)


def test_validate_rejects_non_numeric(valid_frame):
    valid_frame["house_age_years"] = ["old", "new"]
    with pytest.raises(ValueError, match="numeric"):
        validate_dataset(valid_frame)


@pytest.mark.network
def test_load_dataset_end_to_end():
    df = load_dataset()
    assert df.shape == (414, 7)
    assert list(df.columns) == list(config.COLUMN_RENAMES.values())
    assert df[config.TARGET_COLUMN].between(0, 200).all()
