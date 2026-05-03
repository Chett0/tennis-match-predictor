import logging

logger = logging.getLogger(__name__)

import pandas as pd
import glob

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"


def load_data() -> pd.DataFrame:

    logger.info("Loading data from raw files")
    
    files = list(RAW_DATA_PATH.glob("*.xlsx"))

    dfs : list[pd.DataFrame] = []

    for file in files:
        df : pd.DataFrame = pd.read_excel(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Loaded {len(df)} rows from {len(files)} files.")

    return df


def train_test_split(df : pd.DataFrame, test_year : int | None = None, test_size : float = 0.2):

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