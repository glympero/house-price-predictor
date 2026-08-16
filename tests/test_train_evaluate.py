"""End-to-end tests for grouped training, tuning, and final evaluation."""

import json

import numpy as np
import pandas as pd
import pytest
from scripts.post_selection_diagnostics import run_nested_cv_comparison
from sklearn.linear_model import Ridge

from house_prices import config
from house_prices.data import load_dataset
from house_prices.evaluate import evaluate, load_artifact
from house_prices.train import (
    Candidate,
    default_candidates,
    grouped_cv,
    grouped_oof_predictions,
    location_groups,
    select_model,
    split_data,
    train_and_persist,
)

FAST_CANDIDATES = [
    Candidate(
        name="ridge",
        estimator=Ridge(),
        scale=True,
        complexity=1,
        purpose="Fast grouped-search candidate for the end-to-end test.",
        param_grid={"alpha": (0.1, 1.0)},
    )
]


def _comparison(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": name, "complexity": complexity, "cv_rmse": cv_rmse}
            for name, complexity, cv_rmse in rows
        ]
    )


def test_select_model_uses_lowest_rmse_even_when_difference_is_small():
    comparison = _comparison(
        [
            ("mean_baseline", 0, 13.6),
            ("simple", 1, 8.28),
            ("complex", 4, 8.00),
        ]
    )
    chosen, reason = select_model(comparison)
    assert chosen == "complex"
    assert "lowest mean grouped-CV RMSE" in reason
    assert "secondary properties" in reason


def test_tree_candidates_avoid_redundant_transforms_and_scaling():
    candidates = {candidate.name: candidate for candidate in default_candidates()}

    for name in ("random_forest", "gradient_boosting"):
        candidate = candidates[name]
        assert candidate.scale is False
        assert candidate.features == {"log_mrt_distance": False, "age_squared": False}
        assert "scale" not in candidate.build().named_steps

    ridge = candidates["ridge_engineered"]
    assert ridge.scale is True
    assert ridge.features == {"log_mrt_distance": True, "age_squared": True}
    assert "scale" in ridge.build().named_steps


def test_select_model_never_returns_the_baseline():
    comparison = _comparison([("mean_baseline", 0, 8.0), ("simple", 1, 8.1)])
    assert select_model(comparison)[0] == "simple"


def test_nested_cv_reports_outer_scores_without_changing_selection_data(dataset):
    X_train, _, y_train, _ = split_data(dataset)
    groups = location_groups(X_train)
    rows = run_nested_cv_comparison(
        X_train,
        y_train,
        candidates=FAST_CANDIDATES,
        groups=groups,
    )

    assert len(rows) == 1
    assert rows[0]["candidate"] == "ridge"
    assert rows[0]["grid_cells"] == 2
    assert len(rows[0]["outer_fold_rmse"]) == config.CV_FOLDS
    assert len(rows[0]["best_params_by_outer_fold"]) == config.CV_FOLDS
    assert rows[0]["nested_cv_rmse"] > 0


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def trained_dir(dataset, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    train_and_persist(dataset, models_dir, candidates=FAST_CANDIDATES)
    return models_dir


@pytest.mark.network
def test_split_is_reproducible_location_disjoint_and_approximately_80_20(dataset):
    first = split_data(dataset)
    second = split_data(dataset)
    X_train, X_holdout, _, _ = first

    assert X_train.index.equals(second[0].index)
    assert X_holdout.index.equals(second[1].index)
    assert len(X_train) == 331
    assert len(X_holdout) == 83

    train_locations = set(X_train[["latitude", "longitude"]].itertuples(index=False, name=None))
    holdout_locations = set(X_holdout[["latitude", "longitude"]].itertuples(index=False, name=None))
    assert train_locations.isdisjoint(holdout_locations)


@pytest.mark.network
def test_every_grouped_cv_fold_keeps_coordinates_together(dataset):
    X_train, _, y_train, _ = split_data(dataset)
    groups = location_groups(X_train)
    for train_indices, validation_indices in grouped_cv().split(X_train, y_train, groups):
        assert set(groups[train_indices]).isdisjoint(groups[validation_indices])


@pytest.mark.network
def test_training_writes_grouped_tuning_metadata(trained_dir):
    assert (trained_dir / config.MODEL_FILENAME).exists()
    metadata = json.loads((trained_dir / config.METADATA_FILENAME).read_text())
    assert metadata["model"] == "ridge"
    assert metadata["selection_metric"] == "grouped_cv_rmse"
    assert metadata["best_params"]["alpha"] in {0.1, 1.0}
    assert metadata["cv_r2"] > 0.5
    assert metadata["n_location_overlap"] == 0
    assert metadata["n_training_locations"] + metadata["n_holdout_locations"] == 259
    assert len(metadata["cv_comparison"]) == len(FAST_CANDIDATES)
    assert metadata["cv_comparison"][0]["tuning_configurations"] == 2
    assert metadata["data_sha256"] == config.RAW_DATASET_SHA256
    assert metadata["validation_diagnostics"]["largest_errors"]


@pytest.mark.network
def test_artifact_predicts_ordered_intervals(trained_dir, dataset):
    artifact = load_artifact(trained_dir)
    _, X_holdout, _, _ = split_data(dataset)
    point = artifact["point"].predict(X_holdout)
    bounds = artifact["residual_bounds"]

    assert bounds["lower"] < bounds["upper"]
    assert (point + bounds["lower"] <= point + bounds["upper"]).all()
    assert point.min() > 0


@pytest.mark.network
def test_stored_bounds_are_grouped_out_of_fold_residual_percentiles(trained_dir, dataset):
    artifact = load_artifact(trained_dir)
    X_train, _, y_train, _ = split_data(dataset)
    groups = location_groups(X_train)
    out_of_fold = grouped_oof_predictions(artifact["point"], X_train, y_train, groups)
    residuals = y_train.to_numpy() - out_of_fold
    low_q, high_q = config.QUANTILES

    bounds = artifact["residual_bounds"]
    assert bounds["lower"] == pytest.approx(np.percentile(residuals, 100 * low_q))
    assert bounds["upper"] == pytest.approx(np.percentile(residuals, 100 * high_q))


@pytest.mark.network
def test_evaluation_applies_the_stored_offsets(trained_dir, dataset, tmp_path):
    artifact = load_artifact(trained_dir)
    _, X_holdout, _, y_holdout = split_data(dataset)
    bounds = artifact["residual_bounds"]
    predictions = artifact["point"].predict(X_holdout)

    expected_coverage = float(
        (
            (y_holdout.to_numpy() >= predictions + bounds["lower"])
            & (y_holdout.to_numpy() <= predictions + bounds["upper"])
        ).mean()
    )
    expected_width = float(bounds["upper"] - bounds["lower"])

    results = evaluate(dataset, trained_dir, tmp_path)
    assert results["interval_coverage"] == pytest.approx(expected_coverage)
    assert results["interval_mean_width"] == pytest.approx(expected_width)
    assert results["n_location_overlap"] == 0


@pytest.mark.network
def test_evaluation_metrics_uncertainty_and_figures(trained_dir, dataset, tmp_path):
    results = evaluate(dataset, trained_dir, tmp_path)
    assert results["r2"] > 0.4
    assert results["rmse"] < 12
    assert results["n_holdout_rows"] == 83
    assert 0.5 < results["interval_coverage"] <= 1.0
    lower, upper = results["rmse_bootstrap_95_ci"]
    assert 0 < lower < results["rmse"] < upper
    for name in (
        "model_pred_vs_actual.png",
        "model_residuals.png",
        "model_permutation_importance.png",
        "validation_error_by_target_band.png",
    ):
        assert (tmp_path / name).exists()
    assert (trained_dir / config.EVALUATION_FILENAME).exists()
