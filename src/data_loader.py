import logging

logger = logging.getLogger(__name__)

import pandas as pd
import glob

RAW_DATA_PATH = "../data/raw/"
PROCESSED_DATA_PATH = "../data/processed/"


def load_data() -> pd.DataFrame:

    logger.info("Loading data from raw files")
    
    files = glob.glob(f'{RAW_DATA_PATH}*.xlsx')

    dfs : list[pd.DataFrame] = []

    for file in files:
        df : pd.DataFrame = pd.read_excel(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    logger.info(f"Loaded {len(df)} rows from {len(files)} files.")

    return df


def train_test_split(df : pd.DataFrame):

    years = df['date'].dt.year.max() - df['date'].dt.year.min()

    train_df = df[df["date"] < "2025-01-01"]
    test_df = df[df["date"] >= "2025-01-01"]

    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    X_train = train_df.drop(columns=["winner"])
    y_train = train_df["winner"]

    X_test = test_df.drop(columns=["winner"])
    y_test = test_df["winner"]

    return X_train, X_test, y_train, y_test, years