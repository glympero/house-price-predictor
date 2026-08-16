"""FastAPI application serving the trained model.

Run with ``make serve`` (or ``uvicorn house_prices.api.main:app``). The demo UI
is served at ``/`` and the OpenAPI documentation at ``/docs``.

The model is loaded once during startup rather than per request. Loading it at
import time instead would slow down the test suite and make the module
impossible to import without a trained artifact on disk.
"""

import json
import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from house_prices import config
from house_prices.api.schemas import (
    PRICE_UNIT,
    HealthResponse,
    ModelInfoResponse,
    PredictionInterval,
    PredictionRequest,
    PredictionResponse,
)
from house_prices.evaluate import load_artifact

logger = logging.getLogger(__name__)

state: dict = {}


def _load_evaluation() -> dict | None:
    path = config.MODELS_DIR / config.EVALUATION_FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.clear()
    try:
        state["artifact"] = load_artifact(config.MODELS_DIR)
        state["evaluation"] = _load_evaluation()
        logger.info("Loaded model %s", state["artifact"]["metadata"]["model"])
    except FileNotFoundError:
        # Liveness remains available, while /ready and /predict return 503.
        logger.warning("No model artifact found. /ready and /predict will return 503.")
    yield
    state.clear()


app = FastAPI(
    title="House Price Predictor",
    description=(
        "Estimates house price of unit area for properties in Sindian District, "
        "New Taipei City, from a model trained on 2012 and 2013 transactions."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


def _require_artifact() -> dict:
    artifact = state.get("artifact")
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run `make train` and restart the service.",
        )
    return artifact


def _reject_outside_model_support(request: PredictionRequest, metadata: dict) -> None:
    """Refuse to predict for inputs the model was never fitted on.

    The alternative would be to answer anyway, or to clip the input to the
    nearest supported value. Answering produces a number with nothing behind
    it, and clipping silently substitutes a different property for the one that
    was asked about, so both are worse than declining.
    """
    ranges = metadata.get("training_feature_ranges", {})
    unsupported = []
    for field, value in request.model_dump().items():
        bounds = ranges.get(field)
        if bounds is None:
            continue
        if value < bounds["min"] or value > bounds["max"]:
            unsupported.append(
                {
                    "field": field,
                    "value": value,
                    "training_min": bounds["min"],
                    "training_max": bounds["max"],
                }
            )

    if unsupported:
        names = ", ".join(item["field"] for item in unsupported)
        raise HTTPException(
            status_code=422,
            detail={
                "error": "outside_model_support",
                "message": (
                    f"No prediction is served for this input. The following fields fall "
                    f"outside the range the model was fitted on: {names}. The input is not "
                    f"clipped to the supported range, because that would answer a different "
                    f"question."
                ),
                "fields": unsupported,
            },
        )


@app.get("/health", response_model=HealthResponse, tags=["operations"])
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="artifact" in state)


@app.get("/ready", response_model=HealthResponse, tags=["operations"])
def ready() -> HealthResponse:
    _require_artifact()
    return HealthResponse(status="ready", model_loaded=True)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["operations"])
def model_info() -> ModelInfoResponse:
    artifact = _require_artifact()
    metadata = artifact["metadata"]
    features = artifact["point"].named_steps["features"].get_feature_names_out()
    return ModelInfoResponse(
        model_name=metadata["model"],
        trained_at=metadata["trained_at"],
        selection_reason=metadata["selection_reason"],
        n_dataset_rows=metadata["n_dataset_rows"],
        n_training_rows=metadata["n_training_rows"],
        n_holdout_rows=metadata["n_holdout_rows"],
        n_dataset_locations=metadata["n_dataset_locations"],
        n_training_locations=metadata["n_training_locations"],
        n_holdout_locations=metadata["n_holdout_locations"],
        n_location_overlap=metadata["n_location_overlap"],
        split_strategy=metadata["split_strategy"],
        cv_strategy=metadata["cv_strategy"],
        best_params=metadata["best_params"],
        cv_train_rmse=metadata["cv_train_rmse"],
        cv_rmse=metadata["cv_rmse"],
        cv_mae=metadata["cv_mae"],
        cv_r2=metadata["cv_r2"],
        holdout=state.get("evaluation"),
        features_used=list(features),
        training_feature_ranges=metadata.get("training_feature_ranges", {}),
        residual_bounds=artifact["residual_bounds"],
        sklearn_version=metadata["sklearn_version"],
        data_sha256=metadata["data_sha256"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    artifact = _require_artifact()
    metadata = artifact["metadata"]
    _reject_outside_model_support(request, metadata)

    # The pipeline derives every engineered feature itself, so the caller sends
    # raw values and the model's feature choices stay an implementation detail.
    frame = pd.DataFrame([request.model_dump()])
    point = float(artifact["point"].predict(frame)[0])

    # Offsets from the selected model's own residual distribution. They are not
    # symmetric around zero, so the range is not centred on the estimate.
    bounds = artifact["residual_bounds"]
    lower = point + bounds["lower"]
    upper = point + bounds["upper"]

    nominal = round(float(config.QUANTILES[1] - config.QUANTILES[0]), 4)
    evaluation = state.get("evaluation") or {}
    observed = evaluation.get("interval_coverage")

    caveat = (
        f"The range is derived from the residual distribution of the selected model, "
        f"targeting {nominal:.0%} coverage."
    )
    if observed is not None:
        holdout_rows = evaluation.get("n_holdout_rows", metadata["n_holdout_rows"])
        caveat += (
            f" Coverage of {observed:.0%} was observed on the protected, "
            f"coordinate-disjoint {holdout_rows}-row holdout. That is a finite-sample "
            f"estimate, not a guarantee for an individual prediction."
        )
    caveat += " The width is the same for every property and does not widen for unusual inputs."

    return PredictionResponse(
        predicted_price=round(point, 1),
        price_unit=PRICE_UNIT,
        interval=PredictionInterval(
            lower=round(lower, 1),
            upper=round(upper, 1),
            nominal_coverage=nominal,
            observed_holdout_coverage=observed,
            caveat=caveat,
        ),
        model_name=metadata["model"],
        model_trained_at=metadata["trained_at"],
    )


@app.get("/", include_in_schema=False)
def demo_ui() -> FileResponse:
    if not config.UI_FILE.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found.")
    return FileResponse(config.UI_FILE)
