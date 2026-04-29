import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from utils import RAW_DATA_PATH, PROCESSED_DATA_PATH

BET_COLS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
RANK_COLS = ["WRank", "LRank"]
POINT_COLS = ["WPts", "LPts"]
GAME_COLS = ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5"]

CAT_COLS = ["Tournament", "Series"]

ROUND_ORDER = [[
    "Round Robin",
    "1st Round",
    "2nd Round",
    "3rd Round",
    "4th Round",
    "Quarterfinals",
    "Semifinals",
    "The Final"
]]

SERIES_ORDER = [[
    "ATP250",
    "ATP500",
    "Masters 1000",
    "Masters Cup",
    "Grand Slam"
]]

INT_COLS = ["Best of", *RANK_COLS, *GAME_COLS]


def load_2025_data() -> pd.DataFrame:
    data : pd.DataFrame = pd.read_excel(f'{RAW_DATA_PATH}2025.xlsx')
    return data


def load_data() -> pd.DataFrame:
    files = glob.glob(f'{RAW_DATA_PATH}*.xlsx')

    dfs : list[pd.DataFrame] = []

    for file in files:
        df : pd.DataFrame = pd.read_excel(file)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df

class DropFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified features"""

    def __init__(self, features_to_drop: list[str]):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop)
    
class RowFilter(BaseEstimator, TransformerMixin):
    """Custom transformer to filter rows based on a condition"""

    def __init__(self, condition):
        self.condition = condition

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.condition(X)]


class AlignTypesTransformer(BaseEstimator, TransformerMixin):
    """Convert selected columns to selected types"""

    def __init__(self, int_columns):
        self.int_columns = int_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.int_columns] = X[self.int_columns].apply(
            pd.to_numeric, errors="coerce"
        ).astype("Int64")

        return X

class BestOfImputer(BaseEstimator, TransformerMixin):
    """Custom transformer to impute missing values in the 'Best of' column"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        missing_best_of = X["Best of"].isna()

        best_of_5 = (
            (X["W4"] != 0) | (X["W5"] != 0) |
            (
                (X["W1"] > X["L1"]) &
                (X["W2"] > X["L2"]) &
                (X["W3"] > X["L3"])
            )
        )

        X.loc[missing_best_of , "Best of"] = np.where(
            best_of_5.loc[missing_best_of],
            5,
            3
        )

        return X


def save_clean_data(df):
    df.to_excel(f'{PROCESSED_DATA_PATH}tennis_matches_clean.xlsx', index=False)

def clean() -> Pipeline:

    df = load_data()

    game_imputer = SimpleImputer(strategy="constant", fill_value=0)
    bet_imputer = SimpleImputer(strategy="constant", fill_value=1)
    rank_imputer = SimpleImputer(strategy="constant", fill_value=df[RANK_COLS].max().max())
    point_imputer = SimpleImputer(strategy="constant", fill_value=df[POINT_COLS].min().min())
    best_of_imputer = BestOfImputer()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    round_encoder = OrdinalEncoder(categories=ROUND_ORDER)
    series_encoder = OrdinalEncoder(categories=SERIES_ORDER)


    preprocessor = ColumnTransformer(
        transformers=[
            ("game_imputer", game_imputer, GAME_COLS),
            ("bet_imputer", bet_imputer, BET_COLS),
            ("rank_imputer", rank_imputer, RANK_COLS),
            ("point_imputer", point_imputer, POINT_COLS),
            ("best_of_imputer", best_of_imputer),
            ("categorical_encoder", categorical_encoder, CAT_COLS),
            ("round_encoder", round_encoder, ["Round"]),
            ("series_encoder", series_encoder, ["Series"])
        ]
    )

    preprocessor_pipeline = Pipeline(steps=[
        ("drop_cols", DropFeatureSelector(["BFEW", "BFEL"])),
        ("remove_walkovers", RowFilter(lambda df: df["Comment"] != "Walkover")),
        ("valid_sets", RowFilter(lambda df: df["Wsets"].notna() & df["Lsets"].notna())),
        ("align_types", AlignTypesTransformer(INT_COLS)),
        ("preprocessor", preprocessor),
    ])

    return preprocessor_pipeline

if __name__ == "__main__":
    clean()