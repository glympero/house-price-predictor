"""Model training: comparison, selection, and artifact persistence.

Run as ``python -m house_prices.train`` (or ``make train``). The flow:

1. Load and validate the dataset, split off the holdout test set.
2. Cross-validate the candidates on the training portion only, using one
   shared 5-fold split so the scores are comparable.
3. Select a model with :func:`select_model`, which takes the best mean CV RMSE
   and prefers a simpler candidate when the difference is small.
4. Refit the selection on the full training set, measure its out-of-fold
   residuals to derive the prediction interval offsets, and persist both as one
   artifact.

The transaction date is excluded from every candidate. It is measured
separately by :func:`run_date_ablation`, because the reason for dropping it is
that callers cannot supply a meaningful value at prediction time, which is an
argument no cross-validation score can settle.
"""

import hashlib
import json
import logging
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict, cross_validate, train_test_split

from house_prices import config
from house_prices.data import download_dataset, load_dataset
from house_prices.features import build_pipeline

logger = logging.getLogger(__name__)

RAW_FEATURES = {"log_mrt_distance": False, "age_squared": False}
ENGINEERED_FEATURES = {"log_mrt_distance": True, "age_squared": True}


@dataclass(frozen=True)
class Candidate:
    """One entry in the model comparison.

    ``complexity`` orders the candidates from simplest to most complex and is
    used by :func:`select_model` to break ties. It encodes a judgment about
    how much work a model costs to explain and maintain, not a measurement.
    """

    name: str
    estimator: BaseEstimator
    scale: bool
    complexity: int
    purpose: str
    features: dict = field(default_factory=lambda: dict(ENGINEERED_FEATURES))

    def build(self, **overrides):
        return build_pipeline(self.estimator, scale=self.scale, **{**self.features, **overrides})


def default_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="mean_baseline",
            estimator=DummyRegressor(strategy="mean"),
            scale=False,
            complexity=0,
            purpose="Minimum useful performance. Any model that cannot beat it is broken.",
            features=dict(RAW_FEATURES),
        ),
        Candidate(
            name="linear_regression",
            estimator=LinearRegression(),
            scale=True,
            complexity=1,
            purpose="Simplest interpretable model, on the raw columns.",
            features=dict(RAW_FEATURES),
        ),
        Candidate(
            name="linear_engineered",
            estimator=LinearRegression(),
            scale=True,
            complexity=2,
            purpose="Tests whether the EDA-driven transforms let a linear model "
            "capture the nonlinear structure.",
        ),
        Candidate(
            name="ridge_engineered",
            estimator=RidgeCV(alphas=np.logspace(-2, 3, 30)),
            scale=True,
            complexity=3,
            purpose="Adds regularization, because several predictors are correlated.",
        ),
        Candidate(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=300, random_state=config.RANDOM_SEED, n_jobs=-1
            ),
            scale=False,
            complexity=4,
            purpose="Captures nonlinearity and interactions without manual feature work.",
        ),
        Candidate(
            name="gradient_boosting",
            estimator=HistGradientBoostingRegressor(random_state=config.RANDOM_SEED),
            scale=False,
            complexity=5,
            purpose="Second nonlinear ensemble, for comparison with the forest.",
        ),
    ]


SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def split_data(df: pd.DataFrame):
    """The single authoritative train/test split, shared by train and evaluate."""
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=config.TEST_SET_FRACTION, random_state=config.RANDOM_SEED
    )


def _cv() -> KFold:
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)


def run_cv_comparison(X_train, y_train, candidates: list[Candidate] | None = None) -> pd.DataFrame:
    """Score every candidate on the same folds of the training data."""
    candidates = candidates if candidates is not None else default_candidates()
    cv = _cv()
    rows = []
    for candidate in candidates:
        scores = cross_validate(candidate.build(), X_train, y_train, cv=cv, scoring=SCORING)
        rows.append(
            {
                "model": candidate.name,
                "cv_rmse": -scores["test_rmse"].mean(),
                "cv_rmse_std": scores["test_rmse"].std(),
                "cv_mae": -scores["test_mae"].mean(),
                "cv_r2": scores["test_r2"].mean(),
                "complexity": candidate.complexity,
                "purpose": candidate.purpose,
            }
        )
        logger.info(
            "%-20s rmse=%.2f (sd %.2f) mae=%.2f r2=%.3f",
            candidate.name,
            rows[-1]["cv_rmse"],
            rows[-1]["cv_rmse_std"],
            rows[-1]["cv_mae"],
            rows[-1]["cv_r2"],
        )
    return pd.DataFrame(rows)


def run_date_ablation(X_train, y_train, candidates: list[Candidate]) -> pd.DataFrame:
    """Score the given candidates with and without the transaction date."""
    cv = _cv()
    rows = []
    for candidate in candidates:
        for include_date in (False, True):
            scores = cross_validate(
                candidate.build(include_transaction_date=include_date),
                X_train,
                y_train,
                cv=cv,
                scoring=SCORING,
            )
            rows.append(
                {
                    "model": candidate.name,
                    "with_transaction_date": include_date,
                    "cv_rmse": -scores["test_rmse"].mean(),
                    "cv_rmse_std": scores["test_rmse"].std(),
                }
            )
    return pd.DataFrame(rows)


# Candidates whose mean CV RMSE is within this fraction of the best score are
# treated as performing similarly. The scores vary by roughly 2 RMSE between
# folds, which is far larger than the gaps between the leading models, so small
# differences in the mean are not treated as decisive on 331 rows.
SIMILAR_PERFORMANCE_MARGIN = 0.05


def select_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Apply the selection rule and return the chosen model and the reason.

    Take the lowest mean CV RMSE. Any candidate within
    ``SIMILAR_PERFORMANCE_MARGIN`` of it counts as performing similarly, and
    among those the least complex is selected. The baseline is never
    selectable.
    """
    ranked = comparison[comparison["model"] != "mean_baseline"]
    best = ranked.loc[ranked["cv_rmse"].idxmin()]
    threshold = float(best["cv_rmse"]) * (1 + SIMILAR_PERFORMANCE_MARGIN)
    similar = ranked[ranked["cv_rmse"] <= threshold]
    chosen = similar.loc[similar["complexity"].idxmin()]

    if chosen["model"] == best["model"]:
        reason = (
            f"{chosen['model']} has the lowest mean CV RMSE ({chosen['cv_rmse']:.2f}) and no "
            f"simpler candidate comes within {SIMILAR_PERFORMANCE_MARGIN:.0%} of it."
        )
    else:
        difference = float(chosen["cv_rmse"]) - float(best["cv_rmse"])
        reason = (
            f"{best['model']} has the lowest mean CV RMSE ({best['cv_rmse']:.2f}) and "
            f"{chosen['model']} is {chosen['cv_rmse']:.2f}, a difference of {difference:.2f} "
            f"RMSE. That is within {SIMILAR_PERFORMANCE_MARGIN:.0%} and is small next to the "
            f"variation between folds, so it is not treated as evidence that the more complex "
            f"model is better. The simpler and more interpretable candidate is selected."
        )
    return str(chosen["model"]), reason


def fit_residual_bounds(winner: Candidate, X_train, y_train) -> dict[str, float]:
    """Measure how wrong the selected model usually is, as an offset pair.

    The prediction interval is built from the selected model's own errors, so
    the point estimate always lies inside its interval. Residuals are taken
    out-of-fold: an in-sample residual understates the error, because the model
    has already seen the row.
    """
    out_of_fold = cross_val_predict(winner.build(), X_train, y_train, cv=_cv())
    residuals = y_train.to_numpy() - out_of_fold
    low_q, high_q = config.QUANTILES
    return {
        "lower": float(np.percentile(residuals, 100 * low_q)),
        "upper": float(np.percentile(residuals, 100 * high_q)),
    }


def train_and_persist(
    df: pd.DataFrame, models_dir: Path, candidates: list[Candidate] | None = None
) -> dict:
    """Full training flow. Returns the metadata dict it persisted."""
    candidates = candidates if candidates is not None else default_candidates()
    X_train, X_test, y_train, y_test = split_data(df)
    logger.info("Train %d rows, holdout %d rows (not used here)", len(X_train), len(X_test))

    comparison = run_cv_comparison(X_train, y_train, candidates)
    winner_name, reason = select_model(comparison)
    logger.info("Selected %s. %s", winner_name, reason)

    winner = next(c for c in candidates if c.name == winner_name)
    winner_row = comparison[comparison["model"] == winner_name].iloc[0]
    point_pipeline = winner.build()
    point_pipeline.fit(X_train, y_train)

    residual_bounds = fit_residual_bounds(winner, X_train, y_train)

    raw_path = download_dataset()
    metadata = {
        "model": winner_name,
        "selection_reason": reason,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_seed": config.RANDOM_SEED,
        "test_fraction": config.TEST_SET_FRACTION,
        "cv_folds": config.CV_FOLDS,
        "quantiles": list(config.QUANTILES),
        "residual_bounds": residual_bounds,
        # The artifact is fitted on the training rows only, so the evaluated
        # model and the served model are the same object. Recording all three
        # counts keeps "trained on 414" out of the documentation.
        "n_dataset_rows": int(len(df)),
        "n_training_rows": int(len(X_train)),
        "n_holdout_rows": int(len(X_test)),
        # Served alongside the model so the API can tell a caller when an input
        # falls outside the range the model was fitted on.
        "training_feature_ranges": {
            column: {"min": float(X_train[column].min()), "max": float(X_train[column].max())}
            for column in X_train.columns
        },
        "data_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "cv_comparison": comparison.to_dict(orient="records"),
        "cv_rmse": float(winner_row["cv_rmse"]),
        "cv_rmse_std": float(winner_row["cv_rmse_std"]),
        "cv_mae": float(winner_row["cv_mae"]),
        "cv_r2": float(winner_row["cv_r2"]),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "point": point_pipeline,
        "residual_bounds": residual_bounds,
        "metadata": metadata,
    }
    joblib.dump(artifact, models_dir / config.MODEL_FILENAME)
    (models_dir / config.METADATA_FILENAME).write_text(json.dumps(metadata, indent=2))
    logger.info("Artifact saved to %s", models_dir / config.MODEL_FILENAME)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_dataset()
    metadata = train_and_persist(df, config.MODELS_DIR)
    logger.info(
        "Done: %s (cv_rmse=%.2f, cv_mae=%.2f, cv_r2=%.3f)",
        metadata["model"],
        metadata["cv_rmse"],
        metadata["cv_mae"],
        metadata["cv_r2"],
    )


if __name__ == "__main__":
    main()
