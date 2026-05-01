import logging

import sklearn
from sklearn.compose import ColumnTransformer
sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from data_loader import PROCESSED_DATA_PATH, load_data
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


BET_COLS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
RANK_COLS = ["WRank", "LRank"]
POINT_COLS = ["WPts", "LPts"]
GAME_COLS = ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5"]
SETS_COLS = ["Wsets", "Lsets"]

CAT_COLS = ["Tournament"]

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

INT_COLS = [*RANK_COLS, *GAME_COLS]


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


def clean_columns_names(df : pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.replace("num__", "").str.replace("cat__", "").str.replace("remainder__", "")
    return df

def clean_data(df : pd.DataFrame) -> pd.DataFrame:

    game_imputer = SimpleImputer(strategy="constant", fill_value=0)
    sets_imputer = SimpleImputer(strategy="constant", fill_value=0)
    bet_imputer = SimpleImputer(strategy="constant", fill_value=1)
    rank_imputer = SimpleImputer(strategy="constant", fill_value=df[RANK_COLS].max().max())
    point_imputer = SimpleImputer(strategy="constant", fill_value=df[POINT_COLS].min().min())
    best_of_imputer = BestOfImputer()

    categorical_encoder = OneHotEncoder(sparse_output=False)
    round_encoder = OrdinalEncoder(categories=ROUND_ORDER)
    series_encoder = OrdinalEncoder(categories=SERIES_ORDER)

    preprocessor = ColumnTransformer(
        transformers=[
            ("game_imputer", game_imputer, GAME_COLS),
            ("sets_imputer", sets_imputer, SETS_COLS),
            ("bet_imputer", bet_imputer, BET_COLS),
            ("rank_imputer", rank_imputer, RANK_COLS),
            ("point_imputer", point_imputer, POINT_COLS),

            ("categorical_encoder", categorical_encoder, CAT_COLS),
            ("round_encoder", round_encoder, ["Round"]),
            ("series_encoder", series_encoder, ["Series"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    # Pipeline completa
    pipeline = Pipeline([
        ("drop_cols", DropFeatureSelector(["BFEW", "BFEL"])),
        ("remove_walkovers", RowFilter(lambda df: df["Comment"] != "Walkover")),
        ("valid_sets", RowFilter(lambda df: df["Wsets"].notna() & df["Lsets"].notna())),
        ("align_types", AlignTypesTransformer(INT_COLS)),
        ("preprocessing", preprocessor),
        ("best_of_imputer", best_of_imputer)
        # ("clean_columns", FunctionTransformer(clean_columns_names))
    ])

    df_clean = pipeline.fit_transform(df)
    df = pd.DataFrame(df_clean)
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
