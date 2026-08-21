import logging

logger = logging.getLogger(__name__)

import pandas as pd
import glob

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


def load_data() -> pd.DataFrame:
    """Load all raw data files, sort them by date and concatenate them into a single DataFrame."""
    logger.info("Loading data from raw files")

    files = sorted(RAW_DATA_PATH.glob("*.xlsx"))
    dfs : list[pd.DataFrame] = []
    for file in files:
        df : pd.DataFrame = pd.read_excel(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("Date", kind="stable", ignore_index=True)

    logger.info(f"Loaded {len(df)} rows from {len(files)} files.")
    return df


def train_test_split(
        df : pd.DataFrame, 
        test_year : int | None = None, 
        test_size : float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the DataFrame into training and testing sets based on year or size.
    
    Args:
        df: the DataFrame to split
        test_year: if provided, all data from this year and later will be in the test set.
        test_size: if test_year is not provided, this fraction of the data will be in the test set.
    Returns:
        X_train, X_test, y_train, y_test: the split data and labels
    """

    if test_year is not None:
        train_df = df[df['date'].dt.year < test_year]
        test_df = df[df['date'].dt.year >= test_year]
    else:
        train_size = int(len(df) * (1 - test_size))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

    y_train = train_df["winner"]
    X_train = train_df.drop(columns=["winner"])

    y_test = test_df["winner"]
    X_test = test_df.drop(columns=["winner"])

    return X_train, X_test, y_train, y_test