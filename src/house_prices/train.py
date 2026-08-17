"""Grouped model comparison, bounded tuning, selection, and persistence.

Run as ``python -m house_prices.train`` (or ``make train``). The flow is:

1. Reserve an approximately 20% holdout made of complete coordinate groups.
2. Compare and tune candidates with shared five-fold ``GroupKFold`` splits on
   the remaining rows only.
3. Apply the predeclared rule: select the non-baseline candidate with the lowest
   mean grouped-CV RMSE.
4. Refit the selected configuration on all training groups, derive empirical
   residual offsets from grouped out-of-fold predictions, and persist one
   pipeline artifact.

The transaction date is excluded from every production candidate. It is
measured separately by :func:`run_date_ablation`, because a current caller
cannot supply a value with the same meaning as a 2012--2013 transaction date.
"""

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
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, cross_val_predict

from house_prices import config
from house_prices.data import dataset_sha256, download_dataset, load_dataset
from house_prices.features import build_pipeline

logger = logging.getLogger(__name__)

RAW_FEATURES = {"log_mrt_distance": False, "age_squared": False}
ENGINEERED_FEATURES = {"log_mrt_distance": True, "age_squared": True}
LOCATION_COLUMNS = ["latitude", "longitude"]


@dataclass(frozen=True)
class Candidate:
    """One model-family entry and its deliberately bounded search space."""

    name: str
    estimator: BaseEstimator
    scale: bool
    complexity: int
    purpose: str
    features: dict = field(default_factory=lambda: dict(ENGINEERED_FEATURES))
    param_grid: dict[str, tuple[object, ...]] = field(default_factory=dict)

    def build(self, **overrides):
        options = {**self.features, **overrides}
        return build_pipeline(clone(self.estimator), scale=self.scale, **options)

    def pipeline_param_grid(self) -> dict[str, list[object]]:
        return {f"model__{name}": list(values) for name, values in self.param_grid.items()}


@dataclass
class ModelSearch:
    """Comparison table plus each best estimator refitted on all training rows."""

    comparison: pd.DataFrame
    estimators: dict[str, BaseEstimator]


def default_candidates() -> list[Candidate]:
    return [
        Candidate(
            name="mean_baseline",
            estimator=DummyRegressor(strategy="mean"),
            scale=False,
            complexity=0,
            purpose="Minimum useful performance; every real model should beat it.",
            features=dict(RAW_FEATURES),
        ),
        Candidate(
            name="linear_regression",
            estimator=LinearRegression(),
            scale=True,
            complexity=1,
            purpose="Simplest interpretable regression baseline on the raw columns.",
            features=dict(RAW_FEATURES),
        ),
        Candidate(
            name="linear_engineered",
            estimator=LinearRegression(),
            scale=True,
            complexity=2,
            purpose="Tests whether EDA-driven transforms capture the observed curvature.",
        ),
        Candidate(
            name="ridge_engineered",
            estimator=Ridge(),
            scale=True,
            complexity=3,
            purpose="Tests whether regularization helps overlapping location predictors.",
            param_grid={"alpha": (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)},
        ),
        Candidate(
            name="random_forest",
            estimator=RandomForestRegressor(
                n_estimators=300,
                random_state=config.RANDOM_SEED,
                n_jobs=1,
            ),
            scale=False,
            complexity=4,
            purpose="Captures nonlinearities and interactions through tree ensembles.",
            features=dict(RAW_FEATURES),
            param_grid={
                "max_depth": (None, 4, 8),
                "min_samples_leaf": (1, 3, 5),
                "max_features": (0.7, 1.0),
            },
        ),
        Candidate(
            name="gradient_boosting",
            estimator=HistGradientBoostingRegressor(
                early_stopping=False,
                random_state=config.RANDOM_SEED,
            ),
            scale=False,
            complexity=5,
            purpose="Provides a second nonlinear ensemble with different bias and variance.",
            features=dict(RAW_FEATURES),
            param_grid={
                "learning_rate": (0.03, 0.1),
                "max_leaf_nodes": (7, 15, 31),
                "min_samples_leaf": (10, 20, 30),
                "l2_regularization": (0.0, 1.0),
            },
        ),
    ]


SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


def location_groups(X: pd.DataFrame) -> np.ndarray:
    """Return stable integer groups for exact latitude/longitude pairs."""
    missing = set(LOCATION_COLUMNS) - set(X.columns)
    if missing:
        raise ValueError(f"Location grouping requires columns: {sorted(missing)}")
    coordinates = pd.MultiIndex.from_frame(X[LOCATION_COLUMNS])
    codes, _ = pd.factorize(coordinates, sort=True)
    return codes


def split_data(df: pd.DataFrame):
    """Create the authoritative reproducible, coordinate-disjoint holdout.

    ``GroupShuffleSplit`` samples groups, so one draw can miss the requested row
    fraction when group sizes vary. We generate a fixed set of label-blind draws
    and choose the one whose row count is closest to 20%. The target values do
    not influence that choice.
    """
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COLUMN]
    groups = location_groups(X)
    splitter = GroupShuffleSplit(
        n_splits=config.GROUP_SPLIT_CANDIDATES,
        test_size=config.TEST_SET_FRACTION,
        random_state=config.RANDOM_SEED,
    )
    target_rows = round(len(X) * config.TEST_SET_FRACTION)
    candidates = splitter.split(X, groups=groups)
    train_indices, holdout_indices = min(
        candidates,
        key=lambda split: (abs(len(split[1]) - target_rows), tuple(split[1])),
    )
    return (
        X.iloc[train_indices].copy(),
        X.iloc[holdout_indices].copy(),
        y.iloc[train_indices].copy(),
        y.iloc[holdout_indices].copy(),
    )


def grouped_cv() -> GroupKFold:
    """The shared shuffled grouped folds used by every comparison and search."""
    return GroupKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED,
    )


def _resolve_groups(X: pd.DataFrame, groups=None) -> np.ndarray:
    resolved = location_groups(X) if groups is None else np.asarray(groups)
    if len(resolved) != len(X):
        raise ValueError("groups must contain one value per training row")
    if len(np.unique(resolved)) < config.CV_FOLDS:
        raise ValueError(f"grouped CV requires at least {config.CV_FOLDS} unique groups")
    return resolved


def _fit_search(
    candidate: Candidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: np.ndarray,
    **feature_overrides,
) -> GridSearchCV:
    search = GridSearchCV(
        estimator=candidate.build(**feature_overrides),
        param_grid=candidate.pipeline_param_grid(),
        scoring=SCORING,
        refit="rmse",
        cv=grouped_cv(),
        # The dataset/grid are small; one process is faster to start, deterministic,
        # and avoids loky resource-tracker noise when this runs inside Jupyter.
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train, groups=groups)
    return search


def _plain_params(params: dict) -> dict:
    plain = {}
    for name, value in params.items():
        key = name.removeprefix("model__")
        plain[key] = value.item() if isinstance(value, np.generic) else value
    return plain


def _comparison_row(candidate: Candidate, search: GridSearchCV) -> dict:
    index = search.best_index_
    results = search.cv_results_
    return {
        "model": candidate.name,
        "cv_train_rmse": -float(results["mean_train_rmse"][index]),
        "cv_rmse": -float(results["mean_test_rmse"][index]),
        "cv_rmse_std": float(results["std_test_rmse"][index]),
        "cv_mae": -float(results["mean_test_mae"][index]),
        "cv_r2": float(results["mean_test_r2"][index]),
        "complexity": candidate.complexity,
        "purpose": candidate.purpose,
        "tuned": bool(candidate.param_grid),
        "tuning_configurations": len(results["params"]),
        "best_params": _plain_params(search.best_params_),
    }


def run_model_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: list[Candidate] | None = None,
    groups=None,
) -> ModelSearch:
    """Compare and tune every candidate on the same grouped training folds."""
    candidates = candidates if candidates is not None else default_candidates()
    resolved_groups = _resolve_groups(X_train, groups)
    rows = []
    estimators = {}
    for candidate in candidates:
        search = _fit_search(candidate, X_train, y_train, resolved_groups)
        row = _comparison_row(candidate, search)
        rows.append(row)
        estimators[candidate.name] = search.best_estimator_
        logger.info(
            "%-20s train_rmse=%.2f cv_rmse=%.2f (sd %.2f) mae=%.2f params=%s",
            candidate.name,
            row["cv_train_rmse"],
            row["cv_rmse"],
            row["cv_rmse_std"],
            row["cv_mae"],
            row["best_params"],
        )
    return ModelSearch(comparison=pd.DataFrame(rows), estimators=estimators)


def run_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: list[Candidate] | None = None,
    groups=None,
) -> pd.DataFrame:
    """Compatibility wrapper returning the grouped, tuned comparison table."""
    return run_model_search(X_train, y_train, candidates, groups).comparison


def run_date_ablation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: list[Candidate],
    groups=None,
) -> pd.DataFrame:
    """Score candidates with and without the historical transaction date."""
    resolved_groups = _resolve_groups(X_train, groups)
    rows = []
    for candidate in candidates:
        for include_date in (False, True):
            search = _fit_search(
                candidate,
                X_train,
                y_train,
                resolved_groups,
                include_transaction_date=include_date,
            )
            row = _comparison_row(candidate, search)
            rows.append(
                {
                    "model": candidate.name,
                    "with_transaction_date": include_date,
                    "cv_rmse": row["cv_rmse"],
                    "cv_rmse_std": row["cv_rmse_std"],
                    "best_params": row["best_params"],
                }
            )
    return pd.DataFrame(rows)


def select_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """Select the non-baseline candidate with the lowest grouped-CV RMSE."""
    ranked = comparison[comparison["model"] != "mean_baseline"]
    if ranked.empty:
        raise ValueError("At least one non-baseline candidate is required")
    chosen = ranked.loc[ranked["cv_rmse"].idxmin()]
    tuned = bool(chosen.get("tuned", False))
    tuning_phrase = " after bounded tuning" if tuned else ""
    reason = (
        f"{chosen['model']} has the lowest mean grouped-CV RMSE "
        f"({chosen['cv_rmse']:.2f}){tuning_phrase}. It is selected by the predefined "
        "primary metric; interpretability and serving cost are secondary properties, "
        "not overrides."
    )
    return str(chosen["model"]), reason


def grouped_oof_predictions(
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups=None,
) -> np.ndarray:
    resolved_groups = _resolve_groups(X_train, groups)
    return cross_val_predict(
        pipeline,
        X_train,
        y_train,
        groups=resolved_groups,
        cv=grouped_cv(),
        n_jobs=1,
    )


def residual_bounds_from_predictions(y_true: pd.Series, predictions: np.ndarray) -> dict:
    residuals = y_true.to_numpy() - predictions
    low_q, high_q = config.QUANTILES
    return {
        "lower": float(np.percentile(residuals, 100 * low_q)),
        "upper": float(np.percentile(residuals, 100 * high_q)),
    }


def fit_residual_bounds(
    pipeline: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups=None,
) -> dict[str, float]:
    """Derive empirical offsets from coordinate-grouped out-of-fold errors."""
    predictions = grouped_oof_predictions(pipeline, X_train, y_train, groups)
    return residual_bounds_from_predictions(y_train, predictions)


def _segment_metrics(frame: pd.DataFrame, segment: str) -> list[dict]:
    rows = []
    for label, group in frame.groupby(segment, observed=True):
        rows.append(
            {
                "segment": str(label),
                "n": int(len(group)),
                "rmse": float(root_mean_squared_error(group["actual"], group["predicted"])),
                "mae": float(mean_absolute_error(group["actual"], group["predicted"])),
            }
        )
    return rows


def validation_diagnostics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    predictions: np.ndarray,
) -> dict:
    """Summarize grouped out-of-fold errors without touching the holdout."""
    frame = X_train.copy()
    frame["actual"] = y_train
    frame["predicted"] = predictions
    frame["absolute_error"] = np.abs(frame["actual"] - frame["predicted"])
    frame["target_band"] = pd.qcut(frame["actual"], q=3, labels=["low", "middle", "high"])
    frame["mrt_distance_band"] = pd.qcut(
        frame["mrt_distance_m"], q=3, labels=["near", "middle", "far"]
    )
    frame["age_band"] = pd.cut(
        frame["house_age_years"],
        bins=[-np.inf, 10, 20, 30, np.inf],
        labels=["0-10", "10-20", "20-30", "30+"],
    )

    largest = frame.nlargest(5, "absolute_error")
    largest_errors = []
    for index, row in largest.iterrows():
        largest_errors.append(
            {
                "row_id": str(index),
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "absolute_error": float(row["absolute_error"]),
                **{column: float(row[column]) for column in config.FEATURE_COLUMNS},
            }
        )

    return {
        "overall": {
            "rmse": float(root_mean_squared_error(y_train, predictions)),
            "mae": float(mean_absolute_error(y_train, predictions)),
            "r2": float(r2_score(y_train, predictions)),
        },
        "by_target_band": _segment_metrics(frame, "target_band"),
        "by_mrt_distance_band": _segment_metrics(frame, "mrt_distance_band"),
        "by_age_band": _segment_metrics(frame, "age_band"),
        "largest_errors": largest_errors,
    }


def _coordinate_set(X: pd.DataFrame) -> set[tuple[float, float]]:
    return set(X[LOCATION_COLUMNS].itertuples(index=False, name=None))


def train_and_persist(
    df: pd.DataFrame,
    models_dir: Path,
    candidates: list[Candidate] | None = None,
) -> dict:
    """Run the complete training-only workflow and persist its selected pipeline."""
    candidates = candidates if candidates is not None else default_candidates()
    X_train, X_holdout, y_train, _ = split_data(df)
    train_groups = location_groups(X_train)
    train_locations = _coordinate_set(X_train)
    holdout_locations = _coordinate_set(X_holdout)
    overlap = train_locations & holdout_locations
    if overlap:
        raise RuntimeError(f"Location-disjoint split failed; overlap: {sorted(overlap)}")

    logger.info(
        "Training %d rows / %d locations; protecting %d rows / %d locations",
        len(X_train),
        len(train_locations),
        len(X_holdout),
        len(holdout_locations),
    )

    search = run_model_search(X_train, y_train, candidates, train_groups)
    comparison = search.comparison
    winner_name, reason = select_model(comparison)
    logger.info("Selected %s. %s", winner_name, reason)

    point_pipeline = search.estimators[winner_name]
    winner_row = comparison[comparison["model"] == winner_name].iloc[0]
    out_of_fold = grouped_oof_predictions(point_pipeline, X_train, y_train, train_groups)
    residual_bounds = residual_bounds_from_predictions(y_train, out_of_fold)

    raw_path = download_dataset()
    metadata = {
        "model": winner_name,
        "selection_reason": reason,
        "selection_metric": "grouped_cv_rmse",
        "best_params": winner_row["best_params"],
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "random_seed": config.RANDOM_SEED,
        "test_fraction": config.TEST_SET_FRACTION,
        "split_strategy": "coordinate-group-disjoint holdout",
        "group_columns": LOCATION_COLUMNS,
        "cv_strategy": "shuffled GroupKFold",
        "cv_folds": config.CV_FOLDS,
        "quantiles": list(config.QUANTILES),
        "residual_bounds": residual_bounds,
        "n_dataset_rows": int(len(df)),
        "n_training_rows": int(len(X_train)),
        "n_holdout_rows": int(len(X_holdout)),
        "n_dataset_locations": int(len(_coordinate_set(df[config.FEATURE_COLUMNS]))),
        "n_training_locations": int(len(train_locations)),
        "n_holdout_locations": int(len(holdout_locations)),
        "n_location_overlap": 0,
        "training_feature_ranges": {
            column: {"min": float(X_train[column].min()), "max": float(X_train[column].max())}
            for column in X_train.columns
        },
        "data_sha256": dataset_sha256(raw_path),
        "expected_data_sha256": config.RAW_DATASET_SHA256,
        "cv_comparison": comparison.to_dict(orient="records"),
        "cv_train_rmse": float(winner_row["cv_train_rmse"]),
        "cv_rmse": float(winner_row["cv_rmse"]),
        "cv_rmse_std": float(winner_row["cv_rmse_std"]),
        "cv_mae": float(winner_row["cv_mae"]),
        "cv_r2": float(winner_row["cv_r2"]),
        "validation_diagnostics": validation_diagnostics(X_train, y_train, out_of_fold),
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "point": point_pipeline,
        "residual_bounds": residual_bounds,
        "metadata": metadata,
    }
    joblib.dump(artifact, models_dir / config.MODEL_FILENAME)
    (models_dir / config.METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("Artifact saved to %s", models_dir / config.MODEL_FILENAME)
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_dataset()
    metadata = train_and_persist(df, config.MODELS_DIR)
    logger.info(
        "Done: %s (grouped_cv_rmse=%.2f, cv_mae=%.2f, cv_r2=%.3f)",
        metadata["model"],
        metadata["cv_rmse"],
        metadata["cv_mae"],
        metadata["cv_r2"],
    )


if __name__ == "__main__":
    main()
