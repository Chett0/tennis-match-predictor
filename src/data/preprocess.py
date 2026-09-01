import argparse
import builtins
import logging

import numpy as np

import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

sklearn.set_config(transform_output="pandas")

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from src.utils.utils import (
    COMMENT_ROW_DROP,
    get_df_from_pipeline,
    GAME_COLS,
    POINT_COLS,
    BET_COLS,
    RANK_COLS,
    CAT_COLS,
    SETS_COLS,
    COMMENT_TYPOS,
    swap_cols,
    ROUND_ORDER,
    SERIES_ORDER
)
from src.data.loader import PROCESSED_DATA_PATH, load_data, train_test_split
from src.data.transformers import DropFeatureSelector, ReplaceValuesTransformer, BestOfImputer 


def filter_rows(
        conditions : list,
        X : pd.DataFrame,
        y : pd.Series
    ) -> None:
    """Filter rows in the DataFrame based on a list of conditions.

    Args:
        conditions: a list of functions that take a DataFrame and return a boolean mask
        X: the DataFrame to filter
        y: the target Series to filter"""
    mask = pd.Series(True, index=X.index)
    for condition in conditions:
        mask &= condition(X)

    drop_index = X.index[~mask]
    X.drop(index=drop_index, inplace=True)
    y.drop(index=drop_index, inplace=True)


def create_labels(df : pd.DataFrame, random_state : int | None = 42):
    """Randomize winner/loser positions into player1/player2 and build the target.

    Args:
        df: raw DataFrame with ``Winner``/``Loser`` columns and the winner/loser stat columns.
        random_state: seed for the winner/loser swap, so the labels are reproducible
            across runs. Pass ``None`` for a different assignment every time.
    """
    rng = np.random.default_rng(random_state)
    swap_mask : np.typing.NDArray[np.bool[builtins.bool]] = rng.random(len(df)) < 0.5
    df['player1'] = np.where(swap_mask, df['Winner'], df['Loser'])
    df['player2'] = np.where(swap_mask, df['Loser'], df['Winner'])

    # Winner will be an identifier column, not a feature: the sequential player statistics need to know who
    # won each past match, and the score cannot be trusted for retirements/walkovers.
    # It will be dropped in FeatureEngineringTransformer.
    df.drop(columns=["Loser"], inplace=True, errors="ignore")

    for col1, col2 in swap_cols:
        new_col1, new_col2 = None, None
        if col1.startswith("W"):
            new_col1, new_col2 = "player1_" + col1[1:], "player2_" + col2[1:]
        elif col1.endswith("W"):
            new_col1, new_col2 = "player1_" + col1[:-1], "player2_" + col2[:-1]

        if new_col1 and new_col2:
            df[new_col1] = np.where(swap_mask, df[col1], df[col2])
            df[new_col2] = np.where(swap_mask, df[col2], df[col1])

    df["winner"] = np.where(swap_mask, 1, 0)

    drop_cols = [col for pair in swap_cols for col in pair]
    df.drop(columns=drop_cols, inplace=True)


def preprocessing_pipeline(X : pd.DataFrame, y : pd.Series) -> Pipeline:

    filter_rows([
        lambda X: ~X["Comment"].isin(COMMENT_ROW_DROP),
        lambda X: X["player1_sets"].notna() & X["player2_sets"].notna()
    ], X, y)

    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)
    bet_imputer = SimpleImputer(strategy="constant", fill_value=2.0)
    rank_imputer = SimpleImputer(strategy="constant", fill_value=X[RANK_COLS].max().max())
    point_imputer = SimpleImputer(strategy="constant", fill_value=X[POINT_COLS].min().min())
    best_of_imputer = BestOfImputer()

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    round_encoder = OrdinalEncoder(
        categories=ROUND_ORDER,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    series_encoder = OrdinalEncoder(
        categories=SERIES_ORDER,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("zero_imputer", zero_imputer, GAME_COLS + SETS_COLS),
            ("bet_imputer", bet_imputer, BET_COLS),
            ("rank_imputer", rank_imputer, RANK_COLS),
            ("point_imputer", point_imputer, POINT_COLS),

            ("categorical_encoder", categorical_encoder, CAT_COLS),
            ("round_encoder", round_encoder, ["Round"]),
            ("series_encoder", series_encoder, ["Series"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("fix_typos", ReplaceValuesTransformer(column="Comment", replacements=COMMENT_TYPOS)),
            ("drop_cols", DropFeatureSelector(features_to_drop=[
                "player1_BFE", "player2_BFE", "player1_SJ", "player2_SJ",
                "player1_LB", "player2_LB", "player1_EX", "player2_EX",
                ])),
            ("preprocessing", preprocessor),
            ("best_of_imputer", best_of_imputer),
        ],
    )

    return pipeline


def save_clean_data(df : pd.DataFrame):
    df.to_excel(PROCESSED_DATA_PATH / "tennis_matches_clean.xlsx", index=False)

def clean_data(save_data : bool = False):
    logger.info("Starting preprocessing pipeline")

    df : pd.DataFrame = load_data()
    create_labels(df)
    X_train, _, y_train, _ = train_test_split(df)
    pipeline = preprocessing_pipeline(X_train, y_train)
    X_train = get_df_from_pipeline(pipeline, X_train)
    if save_data:
        save_clean_data(df)

    logger.info("Preprocessing pipeline completed")

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save preprocessed data to Excel file"
    )

    args = parser.parse_args()
    clean_data(save_data=args.save)
