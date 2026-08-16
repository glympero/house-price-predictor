"""Reproduce model-selection and protected-holdout sensitivity diagnostics.

This script is intentionally separate from training. It does not select, refit, or
overwrite the shipped model. The nested comparison uses development rows only; the
holdout calculations describe the already-consumed final evaluation.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from house_prices import config
from house_prices.data import load_dataset
from house_prices.evaluate import _bootstrap_rmse_interval, load_artifact
from house_prices.train import (
    Candidate,
    _comparison_row,
    _fit_search,
    _plain_params,
    _resolve_groups,
    default_candidates,
    grouped_cv,
    location_groups,
    split_data,
)

DEFAULT_OUTPUT = config.REPO_ROOT / "docs" / "post_selection_diagnostics.json"


def _grid_cells(candidate: Candidate) -> int:
    return math.prod(len(values) for values in candidate.param_grid.values()) or 1


def run_nested_cv_comparison(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    candidates: list[Candidate] | None = None,
    groups=None,
) -> list[dict]:
    """Compare tuning procedures on grouped outer folds never seen by inner search."""
    candidates = candidates if candidates is not None else default_candidates()
    resolved_groups = _resolve_groups(X_train, groups)
    outer_splits = list(grouped_cv().split(X_train, y_train, resolved_groups))
    rows = []

    for candidate in candidates:
        reported_search = _fit_search(candidate, X_train, y_train, resolved_groups)
        reported_row = _comparison_row(candidate, reported_search)
        outer_scores = []
        outer_params = []

        for outer_train_indices, outer_validation_indices in outer_splits:
            search = _fit_search(
                candidate,
                X_train.iloc[outer_train_indices],
                y_train.iloc[outer_train_indices],
                resolved_groups[outer_train_indices],
            )
            predictions = search.best_estimator_.predict(X_train.iloc[outer_validation_indices])
            outer_scores.append(
                float(root_mean_squared_error(y_train.iloc[outer_validation_indices], predictions))
            )
            outer_params.append(_plain_params(search.best_params_))

        nested_mean = float(np.mean(outer_scores))
        rows.append(
            {
                "candidate": candidate.name,
                "grid_cells": _grid_cells(candidate),
                "reported_cv_rmse": float(reported_row["cv_rmse"]),
                "nested_cv_rmse": nested_mean,
                "optimism": nested_mean - float(reported_row["cv_rmse"]),
                "nested_cv_fold_sd": float(np.std(outer_scores)),
                "outer_fold_rmse": outer_scores,
                "best_params_by_outer_fold": outer_params,
            }
        )

    return rows


def holdout_sensitivity(df: pd.DataFrame, models_dir: Path) -> dict:
    """Describe, without deleting, the largest errors in the consumed holdout."""
    X_train, X_holdout, y_train, y_holdout = split_data(df)
    artifact = load_artifact(models_dir)
    predictions = artifact["point"].predict(X_holdout)
    squared_errors = (y_holdout.to_numpy() - predictions) ** 2
    descending = np.argsort(squared_errors)[::-1]
    worst_position = int(descending[0])
    keep_without_one = np.ones(len(y_holdout), dtype=bool)
    keep_without_one[descending[:1]] = False
    keep_without_three = np.ones(len(y_holdout), dtype=bool)
    keep_without_three[descending[:3]] = False

    _, _, old_y_train, old_y_holdout = train_test_split(
        df[config.FEATURE_COLUMNS],
        df[config.TARGET_COLUMN],
        test_size=config.TEST_SET_FRACTION,
        random_state=config.RANDOM_SEED,
    )

    return {
        "official_rmse": float(np.sqrt(np.mean(squared_errors))),
        "rmse_without_largest_error": float(np.sqrt(np.mean(squared_errors[keep_without_one]))),
        "rmse_without_three_largest_errors": float(
            np.sqrt(np.mean(squared_errors[keep_without_three]))
        ),
        "largest_error_share_of_total_squared_error": float(
            squared_errors[worst_position] / squared_errors.sum()
        ),
        "three_largest_errors_share_of_total_squared_error": float(
            squared_errors[descending[:3]].sum() / squared_errors.sum()
        ),
        "largest_error": {
            "source_row_id": str(y_holdout.index[worst_position]),
            "actual": float(y_holdout.iloc[worst_position]),
            "predicted": float(predictions[worst_position]),
            "residual_actual_minus_predicted": float(
                y_holdout.iloc[worst_position] - predictions[worst_position]
            ),
        },
        "development_target_max": float(y_train.max()),
        "holdout_target_max": float(y_holdout.max()),
        "row_bootstrap_rmse_95_interval": _bootstrap_rmse_interval(
            y_holdout.to_numpy(), predictions
        ),
        "row_bootstrap_rmse_95_interval_without_largest_error": (
            _bootstrap_rmse_interval(
                y_holdout.to_numpy()[keep_without_one], predictions[keep_without_one]
            )
        ),
        "old_random_row_split": {
            "largest_target_was_in_training": bool(
                df[config.TARGET_COLUMN].idxmax() in old_y_train.index
            ),
            "training_target_max": float(old_y_train.max()),
            "holdout_target_max": float(old_y_holdout.max()),
        },
        "interpretation": (
            "Sensitivity analysis only. All 83 valid rows remain in the official "
            "protected-holdout metrics."
        ),
    }


def build_diagnostics(df: pd.DataFrame, models_dir: Path) -> dict:
    X_train, _, y_train, _ = split_data(df)
    groups = location_groups(X_train)
    comparison = run_nested_cv_comparison(X_train, y_train, groups=groups)
    by_name = {row["candidate"]: row for row in comparison}
    forest = by_name["random_forest"]
    boosting = by_name["gradient_boosting"]
    forest_scores = np.asarray(forest["outer_fold_rmse"])
    boosting_scores = np.asarray(boosting["outer_fold_rmse"])

    return {
        "purpose": "Post-selection robustness diagnostic; it does not reopen selection.",
        "protected_holdout_used_for_nested_cv": False,
        "random_seed": config.RANDOM_SEED,
        "development_rows": int(len(X_train)),
        "development_locations": int(len(np.unique(groups))),
        "outer_cv": "shuffled 5-fold GroupKFold by exact coordinate",
        "inner_cv": "shuffled 5-fold GroupKFold by exact coordinate",
        "score_definition": "unweighted mean of outer-fold RMSE values",
        "model_comparison": comparison,
        "nonlinear_pair": {
            "random_forest_minus_histogram_boosting_mean_rmse": float(
                forest["nested_cv_rmse"] - boosting["nested_cv_rmse"]
            ),
            "random_forest_lower_rmse_outer_folds": int(np.sum(forest_scores < boosting_scores)),
            "histogram_boosting_lower_rmse_outer_folds": int(
                np.sum(boosting_scores < forest_scores)
            ),
            "interpretation": (
                "Random forest has the lower mean in this deterministic run, but the "
                "0.14 RMSE gap and 3-to-2 fold split do not separate the families."
            ),
        },
        "protected_holdout_sensitivity": holdout_sensitivity(df, models_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON output path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics = build_diagnostics(load_dataset(), config.MODELS_DIR)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
