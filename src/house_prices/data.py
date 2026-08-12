"""Dataset acquisition and validation.

Downloads the UCI Real Estate Valuation dataset (a small xlsx inside a zip),
caches it under ``data/raw/``, and loads it as a clean, validated DataFrame.
"""

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from house_prices import config

logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Download and cache the raw dataset. Returns the cached xlsx path."""
    target = config.RAW_DATA_DIR / config.RAW_DATASET_FILENAME
    if target.exists() and not force:
        logger.info("Using cached dataset at %s", target)
        return target

    logger.info("Downloading dataset from %s", config.DATASET_URL)
    with urllib.request.urlopen(config.DATASET_URL, timeout=60) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        xlsx_names = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        if not xlsx_names:
            raise RuntimeError(f"No xlsx file found in archive from {config.DATASET_URL}")
        content = archive.read(xlsx_names[0])

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    logger.info("Cached dataset at %s (%d bytes)", target, len(content))
    return target


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset with clean column names, downloading it if needed."""
    path = path or download_dataset()
    df = pd.read_excel(path, index_col=0)

    missing = set(config.COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = df.rename(columns=config.COLUMN_RENAMES)[list(config.COLUMN_RENAMES.values())]
    validate_dataset(df)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise ValueError if the dataframe violates the expected schema."""
    problems: list[str] = []

    expected = list(config.COLUMN_RENAMES.values())
    if list(df.columns) != expected:
        problems.append(f"columns are {list(df.columns)}, expected {expected}")

    if df.empty:
        problems.append("dataframe is empty")

    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_numeric_dtype(series):
            problems.append(f"{column}: expected numeric dtype, got {series.dtype}")
            continue
        nulls = int(series.isna().sum())
        if nulls:
            problems.append(f"{column}: {nulls} null values")
        low, high = config.VALID_RANGES[column]
        out_of_range = int((~series.between(low, high)).sum())
        if out_of_range:
            problems.append(f"{column}: {out_of_range} values outside [{low}, {high}]")

    if problems:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(problems))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_dataset()
    logger.info("Loaded %d rows x %d columns", *df.shape)
    logger.info("Summary:\n%s", df.describe().T)


if __name__ == "__main__":
    main()
