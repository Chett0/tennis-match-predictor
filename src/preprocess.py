import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from data_loader import PROCESSED_DATA_PATH, load_data


BET_COLUMNS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
RANK_COLUMNS = ["WRank", "LRank"]
POINT_COLUMNS = ["WPts", "LPts"]
GAME_COLUMNS = ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5"]


def fill_missing_values(df : pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using defaults."""

    df[BET_COLUMNS] = df[BET_COLUMNS].fillna(1)

    max_rank = max(df["WRank"].max(), df["LRank"].max())
    df[RANK_COLUMNS] = df[RANK_COLUMNS].fillna(max_rank + 100)

    min_points = min(df["WPts"].min(), df["LPts"].min())
    df[POINT_COLUMNS] = df[POINT_COLUMNS].fillna(min_points - 100)

    df[GAME_COLUMNS] = df[GAME_COLUMNS].fillna(0)

    missing_best_of = df["Best of"].isna()

    best_of_5 = (
        (df["W4"] != 0) | (df["W5"] != 0) |
        (
            (df["W1"] > df["L1"]) &
            (df["W2"] > df["L2"]) &
            (df["W3"] > df["L3"])
        )
    )

    df.loc[missing_best_of , "Best of"] = np.where(
        best_of_5.loc[missing_best_of],
        5,
        3
    )

    return df


def remove_data(df : pd.DataFrame) -> pd.DataFrame:
    """Remove unused columns and invalid matches."""

    df.drop(columns=["BFEW", "BFEL"], inplace=True, errors="ignore")
    df = df[df["Comment"] != "Walkover"]
    df = df[df["Wsets"].notna() & df["Lsets"].notna()]

    return df

def categorical_to_numerical(df : pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to numerical."""

    round_order = [[
        "Round Robin",
        "1st Round",
        "2nd Round",
        "3rd Round",
        "4th Round",
        "Quarterfinals",
        "Semifinals",
        "The Final"
    ]]
    round_encoder = OrdinalEncoder(categories=round_order)
    df["Round"] = round_encoder.fit_transform(df[["Round"]]) 

    tournament_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    df["Tournament"] = tournament_encoder.fit_transform(df[["Tournament"]])

    level_order = [[
        "ATP250",
        "ATP500",
        "Masters 1000",
        "Masters Cup",
        "Grand Slam"
    ]]
    level_encoder = OrdinalEncoder(categories=level_order)
    df["Series"] = level_encoder.fit_transform(df[["Series"]])

    return df

def align_types(df : pd.DataFrame) -> pd.DataFrame:
    """Convert columns type."""

    int_columns = ["Best of", *RANK_COLUMNS, *GAME_COLUMNS]
    df[int_columns] = df[int_columns].astype(int)

    return df


def clean_data(df : pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline."""

    df = remove_data(df)
    df = fill_missing_values(df)
    df = categorical_to_numerical(df)
    df = align_types(df)    

    return df

def save_clean_data(df : pd.DataFrame):
    df.to_excel(f'{PROCESSED_DATA_PATH}tennis_matches_clean.xlsx', index=False)

def preprocess_pipeline(df : pd.DataFrame):

    logger.info("Starting preprocessing pipeline")

    df = clean_data(df)
    save_clean_data(df)

    logger.info("Preprocessing pipeline completed")

    return df

if __name__ == "__main__":
    df : pd.DataFrame = load_data()
    preprocess_pipeline(df)