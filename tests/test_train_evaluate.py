"""End-to-end training and evaluation tests on the real dataset.

These use a reduced set of models so the whole flow stays fast. The full
comparison runs via ``make train``.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict

from house_prices import config
from house_prices.data import load_dataset
from house_prices.evaluate import evaluate, load_artifact
from house_prices.train import Candidate, select_model, split_data, train_and_persist

FAST_CANDIDATES = [
    Candidate(
        name="ridge",
        estimator=Ridge(alpha=1.0),
        scale=True,
        complexity=1,
        purpose="Single fast candidate for the end-to-end test.",
    )
]


def _comparison(rows: list[tuple]) -> pd.DataFrame:
    """Build the frame select_model expects from (name, complexity, cv_rmse)."""
    return pd.DataFrame(
        [
            {"model": name, "complexity": complexity, "cv_rmse": cv_rmse}
            for name, complexity, cv_rmse in rows
        ]
    )


def test_select_model_prefers_simpler_model_when_difference_is_small():
    comparison = _comparison(
        [
            ("mean_baseline", 0, 13.6),
            ("simple", 1, 8.28),
            ("complex", 4, 8.00),  # 3.5 percent better, inside the margin
        ]
    )
    chosen, reason = select_model(comparison)
    assert chosen == "simple"
    assert "simpler" in reason


def test_select_model_keeps_complex_model_when_it_is_clearly_better():
    comparison = _comparison(
        [
            ("mean_baseline", 0, 13.6),
            ("simple", 1, 10.2),
            ("complex", 4, 8.0),  # 27 percent better, outside the margin
        ]
    )
    chosen, _ = select_model(comparison)
    assert chosen == "complex"


def test_select_model_never_returns_the_baseline():
    comparison = _comparison([("mean_baseline", 0, 8.0), ("simple", 1, 8.0)])
    assert select_model(comparison)[0] == "simple"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


@pytest.fixture(scope="module")
def trained_dir(dataset, tmp_path_factory):
    models_dir = tmp_path_factory.mktemp("models")
    train_and_persist(dataset, models_dir, candidates=FAST_CANDIDATES)
    return models_dir


@pytest.mark.network
def test_split_is_reproducible(dataset):
    first = split_data(dataset)
    second = split_data(dataset)
    assert first[0].index.equals(second[0].index)
    assert first[1].index.equals(second[1].index)


@pytest.mark.network
def test_training_writes_artifact_and_metadata(trained_dir):
    assert (trained_dir / config.MODEL_FILENAME).exists()
    metadata = json.loads((trained_dir / config.METADATA_FILENAME).read_text())
    assert metadata["model"] == "ridge"
    assert metadata["cv_r2"] > 0.5
    assert len(metadata["cv_comparison"]) == len(FAST_CANDIDATES)
    assert metadata["selection_reason"]
    assert metadata["data_sha256"]


@pytest.mark.network
def test_artifact_predicts_ordered_intervals(trained_dir, dataset):
    artifact = load_artifact(trained_dir)
    _, X_test, _, _ = split_data(dataset)
    point = artifact["point"].predict(X_test)
    bounds = artifact["residual_bounds"]

    assert bounds["lower"] < bounds["upper"]
    assert (point + bounds["lower"] <= point + bounds["upper"]).all()
    assert point.min() > 0


@pytest.mark.network
def test_stored_bounds_are_the_out_of_fold_residual_percentiles(trained_dir, dataset):
    """Recomputing the out-of-fold residuals must reproduce the stored offsets.

    Cross-validation is seeded, so this is deterministic. The test pins both the
    values and the fact that they come from out-of-fold predictions rather than
    from residuals on rows the model has already seen.
    """
    artifact = load_artifact(trained_dir)
    X_train, _, y_train, _ = split_data(dataset)

    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    out_of_fold = cross_val_predict(FAST_CANDIDATES[0].build(), X_train, y_train, cv=cv)
    residuals = y_train.to_numpy() - out_of_fold
    low_q, high_q = config.QUANTILES

    bounds = artifact["residual_bounds"]
    assert bounds["lower"] == pytest.approx(np.percentile(residuals, 100 * low_q))
    assert bounds["upper"] == pytest.approx(np.percentile(residuals, 100 * high_q))


@pytest.mark.network
def test_evaluation_applies_the_stored_offsets(trained_dir, dataset, tmp_path):
    """Coverage must be measured against point + stored offsets, nothing else."""
    artifact = load_artifact(trained_dir)
    _, X_test, _, y_test = split_data(dataset)
    bounds = artifact["residual_bounds"]
    predictions = artifact["point"].predict(X_test)

    expected_coverage = float(
        (
            (y_test.to_numpy() >= predictions + bounds["lower"])
            & (y_test.to_numpy() <= predictions + bounds["upper"])
        ).mean()
    )
    expected_width = float(bounds["upper"] - bounds["lower"])

    results = evaluate(dataset, trained_dir, tmp_path)
    assert results["interval_coverage"] == pytest.approx(expected_coverage)
    assert results["interval_mean_width"] == pytest.approx(expected_width)


@pytest.mark.network
def test_evaluation_metrics_and_figures(trained_dir, dataset, tmp_path):
    results = evaluate(dataset, trained_dir, tmp_path)
    assert results["r2"] > 0.5
    assert results["rmse"] < 10
    assert 0.5 < results["interval_coverage"] <= 1.0
    for name in (
        "model_pred_vs_actual.png",
        "model_residuals.png",
        "model_permutation_importance.png",
    ):
        assert (tmp_path / name).exists()
    assert (trained_dir / config.EVALUATION_FILENAME).exists()
