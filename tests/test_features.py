"""Tests for the feature engineering transformer and pipeline assembly."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from house_prices.features import FeatureEngineer, build_pipeline


@pytest.fixture
def raw_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transaction_date": [2012.9, 2013.2, 2013.5],
            "house_age_years": [5.0, 20.0, 35.0],
            "mrt_distance_m": [100.0, 1000.0, 4000.0],
            "n_convenience_stores": [8, 4, 1],
            "latitude": [24.98, 24.96, 24.95],
            "longitude": [121.54, 121.53, 121.50],
        }
    )


def test_log_transform_applied(raw_features):
    result = FeatureEngineer().fit_transform(raw_features)
    assert "mrt_distance_m" not in result.columns
    assert list(result["log10_mrt_distance"]) == [2.0, 3.0, np.log10(4000)]


def test_raw_distance_kept_when_log_disabled(raw_features):
    result = FeatureEngineer(log_mrt_distance=False).fit_transform(raw_features)
    assert "log10_mrt_distance" not in result.columns
    assert list(result["mrt_distance_m"]) == [100.0, 1000.0, 4000.0]


def test_age_squared_on_by_default(raw_features):
    result = FeatureEngineer().fit_transform(raw_features)
    assert list(result["house_age_squared"]) == [25.0, 400.0, 1225.0]


def test_age_squared_can_be_disabled(raw_features):
    result = FeatureEngineer(age_squared=False).fit_transform(raw_features)
    assert "house_age_squared" not in result.columns


@pytest.mark.parametrize("log_mrt_distance", [True, False])
@pytest.mark.parametrize("age_squared", [True, False])
@pytest.mark.parametrize("include_transaction_date", [True, False])
def test_feature_names_match_produced_columns(
    raw_features, log_mrt_distance, age_squared, include_transaction_date
):
    engineer = FeatureEngineer(
        log_mrt_distance=log_mrt_distance,
        age_squared=age_squared,
        include_transaction_date=include_transaction_date,
    )
    result = engineer.fit_transform(raw_features)
    assert list(engineer.get_feature_names_out()) == list(result.columns)


def test_transaction_date_dropped_by_default(raw_features):
    assert "transaction_date" not in FeatureEngineer().fit_transform(raw_features).columns


def test_transaction_date_kept_when_requested(raw_features):
    result = FeatureEngineer(include_transaction_date=True).fit_transform(raw_features)
    assert "transaction_date" in result.columns


def test_transform_produces_no_missing_values(raw_features):
    result = FeatureEngineer().fit_transform(raw_features)
    assert not result.isna().any().any()


def test_pipeline_with_scaling_fits_and_predicts(raw_features):
    y = pd.Series([50.0, 30.0, 15.0])
    pipeline = build_pipeline(LinearRegression(), scale=True)
    pipeline.fit(raw_features, y)
    predictions = pipeline.predict(raw_features)
    assert predictions.shape == (3,)
    assert np.isfinite(predictions).all()


def test_pipeline_forwards_feature_options(raw_features):
    pipeline = build_pipeline(LinearRegression(), scale=False, age_squared=False)
    transformed = pipeline.named_steps["features"].fit_transform(raw_features)
    assert "house_age_squared" not in transformed.columns


def test_pipeline_accepts_raw_columns_at_inference(raw_features):
    """A caller supplies the original columns, not engineered ones."""
    y = pd.Series([50.0, 30.0, 15.0])
    pipeline = build_pipeline(LinearRegression(), scale=True).fit(raw_features, y)
    one_row = raw_features.iloc[[0]]
    assert pipeline.predict(one_row).shape == (1,)


def test_transform_is_identical_at_training_and_inference(raw_features):
    """The same input must produce the same features however it is passed."""
    engineer = FeatureEngineer().fit(raw_features)
    whole = engineer.transform(raw_features)
    row_by_row = pd.concat([engineer.transform(raw_features.iloc[[i]]) for i in range(3)])
    pd.testing.assert_frame_equal(whole, row_by_row)


def test_scaler_is_fitted_on_training_rows_only(raw_features):
    """Preprocessing must not see rows that are held out."""
    y = pd.Series([50.0, 30.0, 15.0])
    training = raw_features.iloc[:2]
    pipeline = build_pipeline(LinearRegression(), scale=True).fit(training, y.iloc[:2])
    engineered_training = FeatureEngineer().fit_transform(training)
    assert pipeline.named_steps["scale"].mean_ == pytest.approx(
        engineered_training.mean().to_numpy()
    )


def test_serialization_round_trip_preserves_predictions(raw_features, tmp_path):
    y = pd.Series([50.0, 30.0, 15.0])
    pipeline = build_pipeline(LinearRegression(), scale=True).fit(raw_features, y)
    before = pipeline.predict(raw_features)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    after = joblib.load(path).predict(raw_features)
    np.testing.assert_allclose(before, after)
