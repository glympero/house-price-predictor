"""Tests for the demo page.

These are static checks against the served HTML plus real calls to the API. No
browser is involved, so they cannot catch a rendering problem. What they can
catch is the page drifting away from the service: a field the API no longer
accepts, or a demo scenario that stops behaving the way its label claims.

That last one matters most. The scenarios are hard-coded inputs, and the ranges
they sit inside come from whichever rows the model was fitted on. Retrain on
different data and a scenario advertised as answerable can silently become one
the service refuses.
"""

import json

import pytest
from fastapi.testclient import TestClient

from house_prices.api.main import app

FIELDS = [
    "house_age_years",
    "mrt_distance_m",
    "n_convenience_stores",
    "latitude",
    "longitude",
]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def page(client):
    response = client.get("/")
    assert response.status_code == 200
    return response.text


def _extract_presets(html: str) -> list[dict]:
    """Read the PRESETS array out of the page.

    It is written as strict JSON in the page for exactly this reason, so the
    tests read the same values the browser does instead of a copy that can go
    stale.
    """
    marker = "const PRESETS = "
    start = html.index(marker) + len(marker)
    depth = 0
    for offset, character in enumerate(html[start:]):
        if character == "[":
            depth += 1
        elif character == "]":
            depth -= 1
            if depth == 0:
                return json.loads(html[start : start + offset + 1])
    raise AssertionError("The PRESETS array is not closed.")


@pytest.mark.network
def test_the_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")


@pytest.mark.network
def test_the_form_has_an_input_for_every_field_the_api_accepts(page):
    for field in FIELDS:
        assert f'id="{field}"' in page


@pytest.mark.network
def test_the_page_reads_its_supported_ranges_from_the_service(page):
    # Hard-coding them here would let the page advertise limits the model does
    # not have.
    assert '"/model/info"' in page
    assert "training_feature_ranges" in page


@pytest.mark.network
def test_the_page_handles_the_refusal_response(page):
    assert "outside_model_support" in page
    assert "training_min" in page
    assert "training_max" in page


@pytest.mark.network
def test_the_scenarios_cover_both_a_prediction_and_a_refusal(page):
    presets = _extract_presets(page)
    outcomes = {preset["expect"] for preset in presets}
    assert outcomes == {"prediction", "refusal"}


@pytest.mark.network
def test_every_scenario_still_does_what_its_label_says(client, page):
    expected_status = {"prediction": 200, "refusal": 422}

    for preset in _extract_presets(page):
        assert set(preset["values"]) == set(FIELDS), preset["label"]

        response = client.post("/predict", json=preset["values"])
        assert response.status_code == expected_status[preset["expect"]], (
            f"the {preset['label']!r} scenario returned {response.status_code}"
        )

        if preset["expect"] == "refusal":
            assert response.json()["detail"]["error"] == "outside_model_support"
        else:
            assert response.json()["predicted_price"] > 0


@pytest.mark.network
def test_the_edge_scenario_sits_exactly_on_the_training_boundary(client, page):
    """The point of that scenario is that a boundary value is answered."""
    ranges = client.get("/model/info").json()["training_feature_ranges"]
    edge = next(p for p in _extract_presets(page) if p["label"] == "At the edge of the data")

    assert edge["values"]["house_age_years"] == ranges["house_age_years"]["max"]
    assert edge["values"]["mrt_distance_m"] == ranges["mrt_distance_m"]["max"]
