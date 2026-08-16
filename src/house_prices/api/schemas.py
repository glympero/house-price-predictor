"""Request and response models for the prediction API.

The request accepts the raw columns a caller can reasonably know about a
property. It deliberately does not accept ``transaction_date``: the model does
not use it, and asking for a value the caller cannot supply meaningfully would
put the burden of a modelling decision on the client.

Two layers of rejection, both returning 422.

Field bounds reject values that are physically impossible or clearly outside
the district the model describes.

Beyond that, the service refuses to predict for any input that falls outside
the range the model was actually fitted on. A model asked to extrapolate
returns a number with no support behind it, and the input is not clipped to the
nearest supported value either, because that would answer a question the caller
did not ask.
"""

from pydantic import BaseModel, Field

PRICE_UNIT = "10,000 TWD per ping (1 ping = 3.3 m2)"


class PredictionRequest(BaseModel):
    house_age_years: float = Field(
        ge=0, le=100, description="Age of the property in years.", examples=[10.0]
    )
    mrt_distance_m: float = Field(
        gt=0,
        le=10_000,
        description="Straight-line distance to the nearest MRT station, in metres.",
        examples=[250.0],
    )
    n_convenience_stores: int = Field(
        ge=0, le=20, description="Convenience stores within walking distance.", examples=[6]
    )
    latitude: float = Field(
        ge=24.8, le=25.2, description="Latitude, within the Taipei region.", examples=[24.975]
    )
    longitude: float = Field(
        ge=121.4, le=121.7, description="Longitude, within the Taipei region.", examples=[121.540]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "house_age_years": 10.0,
                    "mrt_distance_m": 250.0,
                    "n_convenience_stores": 6,
                    "latitude": 24.975,
                    "longitude": 121.540,
                }
            ]
        }
    }


class PredictionInterval(BaseModel):
    lower: float = Field(description="Lower bound of the estimated range.")
    upper: float = Field(description="Upper bound of the estimated range.")
    nominal_coverage: float = Field(
        description="Coverage the interval was designed for, as a fraction."
    )
    observed_holdout_coverage: float | None = Field(
        default=None,
        description=(
            "Coverage observed on the protected coordinate-disjoint holdout. "
            "It is a finite-sample estimate, not a per-prediction guarantee."
        ),
    )
    caveat: str = Field(description="Plain statement of how far the interval can be trusted.")


class PredictionResponse(BaseModel):
    predicted_price: float = Field(description=f"Point estimate, in {PRICE_UNIT}.")
    price_unit: str = PRICE_UNIT
    interval: PredictionInterval
    model_name: str
    model_trained_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_at: str
    selection_reason: str
    n_dataset_rows: int
    n_training_rows: int
    n_holdout_rows: int
    n_dataset_locations: int
    n_training_locations: int
    n_holdout_locations: int
    n_location_overlap: int
    split_strategy: str
    cv_strategy: str
    best_params: dict
    cv_train_rmse: float
    cv_rmse: float
    cv_mae: float
    cv_r2: float
    holdout: dict | None
    features_used: list[str]
    training_feature_ranges: dict
    residual_bounds: dict
    sklearn_version: str
    data_sha256: str
    price_unit: str = PRICE_UNIT
