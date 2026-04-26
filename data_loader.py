"""
Shared data loading utilities for the Jena Climate time series series.
All weekends import from here to guarantee a consistent preprocessing pipeline.
"""

from pathlib import Path
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "jena_climate_2009_2016.csv"

# ---------------------------------------------------------------------------
# Train / validation / test split boundaries (locked for the whole series)
# ---------------------------------------------------------------------------
TRAIN_START = "2009-01-01"
TRAIN_END = "2014-12-31"
VAL_START = "2015-01-01"
VAL_END = "2015-12-31"
TEST_START = "2016-01-01"
TEST_END = "2016-12-31"

TARGET = "T (degC)"

# Maximum gap to forward-fill (hours)
MAX_FILL_HOURS = 6


def load_hourly(path: Path = RAW_CSV) -> pd.DataFrame:
    """
    Load the raw Jena Climate CSV and return a clean hourly DataFrame.

    Steps
    -----
    1. Parse "Date Time" as a DatetimeIndex.
    2. Drop duplicate timestamps.
    3. Resample to hourly using the mean.
    4. Forward-fill gaps <= MAX_FILL_HOURS.
    5. Truncate to clean year boundaries (2009–2016).

    Returns
    -------
    pd.DataFrame
        Hourly DataFrame with DatetimeIndex, all original columns.
    """
    df = pd.read_csv(path, index_col="Date Time")
    df.index = pd.to_datetime(df.index, dayfirst=True)

    df = df[~df.index.duplicated(keep="first")]

    df = df.resample("h").mean()

    # Forward-fill short gaps only; longer gaps remain NaN
    df = df.ffill(limit=MAX_FILL_HOURS)

    df = df.loc[TRAIN_START:TEST_END]

    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) slices of df using the locked boundaries."""
    train = df.loc[TRAIN_START:TRAIN_END]
    val = df.loc[VAL_START:VAL_END]
    test = df.loc[TEST_START:TEST_END]
    return train, val, test
