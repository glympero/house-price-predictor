"""Central configuration: paths, dataset source, and reproducibility settings."""

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    """Let the deployment decide where things live.

    Inside the repository the layout is the obvious answer, so it stays the
    default. Elsewhere it is not: in a container the model does not have to sit
    next to the source, and mounting a newer one should not require a rebuild.
    Deriving these paths from the location of this file alone would tie the
    data to wherever the package happened to be installed.
    """
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


REPO_ROOT = _path_from_env("HOUSE_PRICES_ROOT", Path(__file__).resolve().parents[2])
DATA_DIR = _path_from_env("HOUSE_PRICES_DATA_DIR", REPO_ROOT / "data")
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = _path_from_env("HOUSE_PRICES_MODELS_DIR", REPO_ROOT / "models")
REPORTS_DIR = _path_from_env("HOUSE_PRICES_REPORTS_DIR", REPO_ROOT / "docs" / "figures")
UI_FILE = _path_from_env("HOUSE_PRICES_UI_FILE", REPO_ROOT / "ui" / "index.html")

DATASET_URL = "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip"
RAW_DATASET_FILENAME = "real_estate_valuation.xlsx"
RAW_DATASET_SHA256 = "597d72fcc6c0539e6035a033ddb387db48fff3fb1f3c98fee31fe081c64a9059"

# Seeds every stochastic step (train/test split, model randomness) so that
# any run reproduces the same split, models, and metrics.
RANDOM_SEED = 42
TEST_SET_FRACTION = 0.2
GROUP_SPLIT_CANDIDATES = 200
CV_FOLDS = 5

# Prediction interval bounds served next to the point estimate.
QUANTILES = (0.05, 0.95)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
EVALUATION_FILENAME = "evaluation.json"

TARGET_COLUMN = "price_per_unit_area"

# Original UCI column headers mapped to clean snake_case names.
COLUMN_RENAMES = {
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age_years",
    "X3 distance to the nearest MRT station": "mrt_distance_m",
    "X4 number of convenience stores": "n_convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": TARGET_COLUMN,
}

FEATURE_COLUMNS = [c for c in COLUMN_RENAMES.values() if c != TARGET_COLUMN]

# Plausibility envelopes for ingestion validation, derived from the dataset's
# documentation (Sindian District, New Taipei City, Taiwan, transactions from
# 2012 and 2013). That is where the coordinate window and the date bounds come
# from. Kept deliberately loose: they should catch corruption and unit errors,
# not reject legitimate values near the observed extremes.
VALID_RANGES = {
    "transaction_date": (2012.0, 2014.0),
    "house_age_years": (0.0, 100.0),
    "mrt_distance_m": (0.0, 10_000.0),
    "n_convenience_stores": (0, 20),
    "latitude": (24.8, 25.2),
    "longitude": (121.4, 121.7),
    TARGET_COLUMN: (0.0, 200.0),
}
