"""Central configuration: paths, dataset source, and reproducibility settings."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = REPO_ROOT / "models"
REPORTS_DIR = REPO_ROOT / "docs" / "figures"

DATASET_URL = "https://archive.ics.uci.edu/static/public/477/real+estate+valuation+data+set.zip"
RAW_DATASET_FILENAME = "real_estate_valuation.xlsx"

# Seeds every stochastic step (train/test split, model randomness) so that
# any run reproduces the same split, models, and metrics.
RANDOM_SEED = 42
TEST_SET_FRACTION = 0.2

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
# documentation (Sindian District, New Taipei City, Taiwan; transactions
# 2012-2013 — hence the coordinate window around the district and the date
# bounds). Kept deliberately loose: they should catch corruption and unit
# errors, not reject legitimate values near the observed extremes.
VALID_RANGES = {
    "transaction_date": (2012.0, 2014.0),
    "house_age_years": (0.0, 100.0),
    "mrt_distance_m": (0.0, 10_000.0),
    "n_convenience_stores": (0, 20),
    "latitude": (24.8, 25.2),
    "longitude": (121.4, 121.7),
    TARGET_COLUMN: (0.0, 200.0),
}
