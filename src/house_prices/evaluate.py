"""Evaluation of the persisted model on the holdout test set.

Run as ``python -m house_prices.evaluate`` after training. Produces the
final metrics (RMSE, MAE, R², interval coverage), the diagnostic figures,
and ``models/evaluation.json``.
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    """Score the artifact on the holdout set and write metrics + figures."""
    artifact = load_artifact(models_dir)
    _, X_test, _, y_test = split_data(df)

    predictions = artifact["point"].predict(X_test)
    bounds = artifact["residual_bounds"]
    lower = predictions + bounds["lower"]
    upper = predictions + bounds["upper"]

    results = {
        "model": artifact["metadata"]["model"],
        "n_test_rows": int(len(y_test)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
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

    (models_dir / config.EVALUATION_FILENAME).write_text(json.dumps(results, indent=2))
    return results


def _plot_predictions(y_test, predictions, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, predictions, s=22, alpha=0.6)
    lims = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
    ax.plot(lims, lims, "--", color="gray", label="perfect prediction")
    ax.set(xlabel="actual price", ylabel="predicted price", title="Holdout: predicted vs actual")
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
        title="Holdout residuals",
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
        title="Permutation importance on the holdout set",
    )
    fig.savefig(figures_dir / "model_permutation_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_dataset()
    evaluate(df, config.MODELS_DIR, config.REPORTS_DIR)


if __name__ == "__main__":
    main()
