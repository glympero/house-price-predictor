"""One final evaluation of the persisted model on the protected holdout.

Run as ``python -m house_prices.evaluate`` after training. Produces the
final metrics (RMSE, MAE, R², interval coverage), a bootstrap RMSE interval,
diagnostic figures, and ``models/evaluation.json``.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from house_prices import config
from house_prices.data import load_dataset
from house_prices.train import split_data

logger = logging.getLogger(__name__)


def load_artifact(models_dir: Path) -> dict:
    path = models_dir / config.MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"No model artifact at {path}. Run training first.")
    return joblib.load(path)


def evaluate(df: pd.DataFrame, models_dir: Path, figures_dir: Path) -> dict:
    """Score the frozen artifact on the coordinate-disjoint holdout."""
    artifact = load_artifact(models_dir)
    _, X_test, _, y_test = split_data(df)

    predictions = artifact["point"].predict(X_test)
    bounds = artifact["residual_bounds"]
    lower = predictions + bounds["lower"]
    upper = predictions + bounds["upper"]

    rmse_interval = _bootstrap_rmse_interval(y_test.to_numpy(), predictions)
    metadata = artifact["metadata"]
    results = {
        "model": artifact["metadata"]["model"],
        "evaluation_type": "protected coordinate-group-disjoint holdout",
        "n_holdout_rows": int(len(y_test)),
        "n_holdout_locations": int(metadata["n_holdout_locations"]),
        "n_location_overlap": int(metadata["n_location_overlap"]),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "rmse_bootstrap_95_ci": rmse_interval,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "interval_coverage": float(((y_test >= lower) & (y_test <= upper)).mean()),
        "interval_mean_width": float((upper - lower).mean()),
    }
    logger.info(
        "Holdout: rmse=%.2f mae=%.2f r2=%.3f coverage=%.0f%% (target %d%%)",
        results["rmse"],
        results["mae"],
        results["r2"],
        100 * results["interval_coverage"],
        round(100 * (config.QUANTILES[1] - config.QUANTILES[0])),
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_predictions(y_test, predictions, figures_dir)
    _plot_residuals(y_test, predictions, figures_dir)
    _plot_importance(artifact["point"], X_test, y_test, figures_dir)
    _plot_validation_segments(metadata["validation_diagnostics"], figures_dir)

    (models_dir / config.EVALUATION_FILENAME).write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def _bootstrap_rmse_interval(
    y_true: np.ndarray,
    predictions: np.ndarray,
    *,
    repetitions: int = 2000,
) -> list[float]:
    """Deterministic row-bootstrap interval for the holdout RMSE."""
    squared_errors = (y_true - predictions) ** 2
    rng = np.random.default_rng(config.RANDOM_SEED)
    indices = rng.integers(0, len(y_true), size=(repetitions, len(y_true)))
    bootstrapped = np.sqrt(squared_errors[indices].mean(axis=1))
    lower, upper = np.percentile(bootstrapped, [2.5, 97.5])
    return [float(lower), float(upper)]


def _plot_predictions(y_test, predictions, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, predictions, s=22, alpha=0.6)
    lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(lims, lims, "--", color="gray", label="perfect prediction")
    ax.set(
        xlabel="actual price per unit area",
        ylabel="predicted price per unit area",
        title="Protected location holdout: predicted vs actual",
    )
    ax.legend()
    fig.savefig(figures_dir / "model_pred_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residuals(y_test, predictions, figures_dir: Path) -> None:
    residuals = y_test - predictions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(predictions, residuals, s=22, alpha=0.6)
    ax.axhline(0, color="gray", ls="--")
    ax.set(
        xlabel="predicted price",
        ylabel="residual (actual minus predicted)",
        title="Protected location holdout: residuals",
    )
    fig.savefig(figures_dir / "model_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_importance(pipeline, X_test, y_test, figures_dir: Path) -> None:
    importance = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=config.RANDOM_SEED,
        scoring="neg_root_mean_squared_error",
    )
    order = importance.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(
        [X_test.columns[i] for i in order],
        importance.importances_mean[order],
        xerr=importance.importances_std[order],
        color="#4878cf",
    )
    ax.set(
        xlabel="RMSE increase when the feature is shuffled",
        title="Permutation importance on the protected holdout",
    )
    fig.savefig(figures_dir / "model_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_validation_segments(diagnostics: dict, figures_dir: Path) -> None:
    rows = diagnostics["by_target_band"]
    labels = [row["segment"] for row in rows]
    rmse = [row["rmse"] for row in rows]
    mae = [row["mae"] for row in rows]
    positions = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(positions - width / 2, rmse, width, label="RMSE", color="#087E8B")
    ax.bar(positions + width / 2, mae, width, label="MAE", color="#FF5A5F")
    ax.set_xticks(positions, labels)
    ax.set(
        xlabel="actual-price band",
        ylabel="error (10,000 TWD per ping)",
        title="Grouped out-of-fold error by target band",
    )
    ax.legend()
    fig.savefig(
        figures_dir / "validation_error_by_target_band.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_dataset()
    evaluate(df, config.MODELS_DIR, config.REPORTS_DIR)


if __name__ == "__main__":
    main()
