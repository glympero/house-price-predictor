"""Tests for the serving layer, driven through the HTTP interface."""

import pytest
from fastapi.testclient import TestClient

from house_prices.api.main import app, state
from house_prices.data import load_dataset
from house_prices.train import split_data

VALID_REQUEST = {
    "house_age_years": 10.0,
    "mrt_distance_m": 250.0,
    "n_convenience_stores": 6,
    "latitude": 24.975,
    "longitude": 121.540,
}


@pytest.fixture(scope="module")
def client():
    # The context manager runs the lifespan, which loads the model.
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.network
def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.network
def test_readiness_requires_a_loaded_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_loaded": True}

    artifact = state.pop("artifact")
    try:
        response = client.get("/ready")
        assert response.status_code == 503
    finally:
        state["artifact"] = artifact


@pytest.mark.network
def test_predict_returns_price_and_ordered_interval(client):
    response = client.post("/predict", json=VALID_REQUEST)
    assert response.status_code == 200
    body = response.json()

    assert body["predicted_price"] > 0
    assert body["interval"]["lower"] < body["interval"]["upper"]
    assert body["price_unit"].startswith("10,000 TWD")
    assert body["model_name"]


@pytest.mark.network
def test_serving_applies_the_stored_residual_offsets(client):
    """The bounds must be the stored offsets applied to this prediction."""
    bounds = client.get("/model/info").json()["residual_bounds"]
    body = client.post("/predict", json=VALID_REQUEST).json()

    # Both the estimate and the bounds are rounded for display, so allow for
    # the rounding rather than asserting exact equality.
    point = body["predicted_price"]
    assert body["interval"]["lower"] == pytest.approx(point + bounds["lower"], abs=0.11)
    assert body["interval"]["upper"] == pytest.approx(point + bounds["upper"], abs=0.11)
    assert body["interval"]["upper"] - body["interval"]["lower"] == pytest.approx(
        bounds["upper"] - bounds["lower"], abs=0.11
    )


@pytest.mark.network
def test_coverage_is_reported_as_finite_sample_holdout_evidence(client):
    interval = client.post("/predict", json=VALID_REQUEST).json()["interval"]
    assert interval["nominal_coverage"] == 0.9
    assert interval["observed_holdout_coverage"] > 0
    assert "coordinate-disjoint" in interval["caveat"]
    assert "not a guarantee" in interval["caveat"]
    assert "same for every property" in interval["caveat"]


@pytest.mark.network
def test_input_outside_model_support_is_rejected(client):
    """The training data reaches 43.8 years, so 95 has no support behind it."""
    payload = {**VALID_REQUEST, "house_age_years": 95.0, "mrt_distance_m": 9000.0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "outside_model_support"
    offending = {item["field"] for item in detail["fields"]}
    assert offending == {"house_age_years", "mrt_distance_m"}


@pytest.mark.network
def test_out_of_support_input_is_not_clipped(client):
    """Rejection must not be quietly replaced by predicting the nearest value."""
    payload = {**VALID_REQUEST, "house_age_years": 90.0}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "predicted_price" not in response.json()


@pytest.mark.network
def test_values_at_the_edge_of_support_are_still_served(client):
    """Rejection applies outside the range, not at its boundary."""
    ranges = client.get("/model/info").json()["training_feature_ranges"]
    payload = {**VALID_REQUEST, "house_age_years": ranges["house_age_years"]["max"]}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize(
    "field, value",
    [
        ("house_age_years", -1.0),
        ("mrt_distance_m", 0.0),
        ("n_convenience_stores", -3),
        ("latitude", 48.0),
        ("longitude", 2.0),
    ],
)
@pytest.mark.network
def test_impossible_values_are_rejected(client, field, value):
    response = client.post("/predict", json={**VALID_REQUEST, field: value})
    assert response.status_code == 422


@pytest.mark.network
def test_unknown_fields_do_not_break_the_contract(client):
    """Extra keys are ignored, so adding one client-side cannot break serving."""
    response = client.post("/predict", json={**VALID_REQUEST, "transaction_date": 2013.5})
    assert response.status_code == 200


@pytest.mark.network
def test_model_info_exposes_provenance(client):
    body = client.get("/model/info").json()
    assert body["model_name"]
    assert body["selection_reason"]
    assert len(body["data_sha256"]) == 64
    assert set(body["features_used"]) == {
        "house_age_years",
        "mrt_distance_m",
        "n_convenience_stores",
        "latitude",
        "longitude",
    }
    assert body["residual_bounds"]["lower"] < body["residual_bounds"]["upper"]
    assert body["split_strategy"] == "coordinate-group-disjoint holdout"
    assert body["cv_strategy"] == "shuffled GroupKFold"
    assert body["best_params"]


@pytest.mark.network
def test_model_info_separates_rows_locations_and_has_no_overlap(client):
    body = client.get("/model/info").json()
    assert body["n_dataset_rows"] == 414
    assert body["n_training_rows"] == 331
    assert body["n_holdout_rows"] == 83
    assert body["n_training_rows"] + body["n_holdout_rows"] == body["n_dataset_rows"]
    assert body["n_training_locations"] == 207
    assert body["n_holdout_locations"] == 52
    assert body["n_location_overlap"] == 0


@pytest.mark.network
def test_training_ranges_come_from_the_fitted_rows(client):
    """Ranges must describe the grouped training rows, not the whole dataset."""
    ranges = client.get("/model/info").json()["training_feature_ranges"]
    X_train, _, _, _ = split_data(load_dataset())
    for column in X_train.columns:
        assert ranges[column]["min"] == pytest.approx(X_train[column].min())
        assert ranges[column]["max"] == pytest.approx(X_train[column].max())


@pytest.mark.network
def test_model_info_reports_protected_holdout_metrics(client):
    holdout = client.get("/model/info").json()["holdout"]
    assert holdout["n_holdout_rows"] == 83
    assert holdout["n_holdout_locations"] == 52
    assert holdout["n_location_overlap"] == 0
    assert 0 < holdout["interval_coverage"] <= 1.0
    assert holdout["rmse"] > 0


@pytest.mark.network
def test_demo_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "House Price Predictor" in response.text


@pytest.mark.network
def test_openapi_documents_the_endpoints(client):
    schema = client.get("/openapi.json").json()
    assert "/predict" in schema["paths"]
    assert "/health" in schema["paths"]
    assert "/ready" in schema["paths"]
    assert "/model/info" in schema["paths"]
