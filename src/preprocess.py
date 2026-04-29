import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from utils import RAW_DATA_PATH, PROCESSED_DATA_PATH

BET_COLS = ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]
RANK_COLS = ["WRank", "LRank"]
POINT_COLS = ["WPts", "LPts"]
GAME_COLS = ["W1", "L1", "W2", "L2", "W3", "L3", "W4", "L4", "W5", "L5"]

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
    
class SimpleImputerCustom(BaseEstimator, TransformerMixin):
    """Custom transformer"""

    def __init__(self, variables, strategy="constant", fill_value=0):
        self.variables = variables
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = SimpleImputer(missing_values=np.nan, strategy=strategy, fill_value=fill_value)

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.imputer.fit(X_)
        return self

    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed = pd.DataFrame(self.imputer.transform(X_), # type: ignore
                         columns=self.variables)
        X.drop(self.variables, axis= 1, inplace=True)
        X[self.variables] = X_transformed[self.variables].values
        return X

class OneHotEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables
        self.ohe = OneHotEncoder(drop='first', 
            handle_unknown = 'ignore')
    def fit(self, X, y = None):
        X_ = X.loc[:,self.variables]
        self.ohe.fit(X_)
        return self
    def transform(self, X):
        X_ = X.loc[:,self.variables]
        X_transformed = pd.DataFrame(self.ohe.transform(X_).toarray(), columns= self.ohe.get_feature_names_out()) # type: ignore
        X_remaining = X.drop(self.variables, axis= 1)
        X = pd.concat([X_remaining, X_transformed], axis=1)
        return X
    
class OrdinalEncoderCustom(BaseEstimator, TransformerMixin):
    def __init__(self, variables, categories):
        self.variables = variables
        self.categories = categories
        self.encoder = OrdinalEncoder(
            categories=categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )

    def fit(self, X, y=None):
        X_ = X.loc[:, self.variables]
        self.encoder.fit(X_)
        return self

    def transform(self, X):
        X = X.copy()
        X_ = X.loc[:, self.variables]

        X_encoded = pd.DataFrame(
            self.encoder.transform(X_),
            columns=self.variables,
            index=X.index
        )

        X.drop(self.variables, axis=1, inplace=True)
        X[self.variables] = X_encoded

        return X

def save_clean_data(df):
    df.to_csv(f'{PROCESSED_DATA_PATH}tennis_matches_clean.csv', index=False)

def clean():

        df = load_data()

        game_imputer = SimpleImputerCustom(variables=GAME_COLS, strategy="constant", fill_value=0)
        bet_imputer = SimpleImputerCustom(variables=BET_COLS, strategy="constant", fill_value=1)
        rank_imputer = SimpleImputerCustom(variables=RANK_COLS, strategy="constant", fill_value=df[RANK_COLS].max().max())
        point_imputer = SimpleImputerCustom(variables=POINT_COLS, strategy="constant", fill_value=df[POINT_COLS].min().min())
        best_of_imputer = BestOfImputer()

        categorical_encoder = OneHotEncoderCustom(variables=CAT_COLS)
        round_encoder = OrdinalEncoderCustom(variables=["Round"], categories=ROUND_ORDER)
        series_encoder = OrdinalEncoderCustom(variables=["Series"], categories=SERIES_ORDER)

        num_pipeline = Pipeline(steps=[
            ("game_imputer", game_imputer),
            ("bet_imputer", bet_imputer),
            ("rank_imputer", rank_imputer),
            ("point_imputer", point_imputer),
            ("best_of_imputer", best_of_imputer)
        ])

        cat_pipeline = Pipeline(steps=[
            ("categorical_encoder", categorical_encoder),
            ("round_encoder", round_encoder),
            ("series_encoder", series_encoder)
        ])

        preprocessor_pipeline = Pipeline(steps=[
            ("drop_cols", DropFeatureSelector(["BFEW", "BFEL"])),
            ("remove_walkovers", RowFilter(lambda df: df["Comment"] != "Walkover")),
            ("valid_sets", RowFilter(lambda df: df["Wsets"].notna() & df["Lsets"].notna())),
            ("align_types", AlignTypesTransformer(INT_COLS)),
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])

        df = preprocessor_pipeline.fit_transform(df)
        df = pd.DataFrame(df)
        save_clean_data(df)
        return df


if __name__ == "__main__":
    clean()