import pandas as pd
import numpy as np

from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances

from src.utils.utils import GAME_COLS


class DropFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified features"""

    def __init__(self, features_to_drop: list[str], is_grouped: bool = False):
        self.features_to_drop = features_to_drop
        self.is_grouped = is_grouped

    def fit(self, X, y=None):
        if self.is_grouped:
            self.features_to_drop_ = [
                col
                for feature in self.features_to_drop
                for col in X.columns
                if col.startswith(feature)
            ]
        else:
            self.features_to_drop_ = list(self.features_to_drop)
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop_, errors="ignore")
    

class ReplaceValuesTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to replace known wrong values in a column with their canonical version"""

    def __init__(self, column : str, replacements : dict[str, str]):
        self.column = column
        self.replacements = replacements

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = X[self.column].replace(self.replacements)
        return X



class RankPointsImputer(BaseEstimator, TransformerMixin):
    """Impute missing player ranks and points from earlier rows."""

    def __init__(self, default_rank : int, default_points : int):
        self.players : dict[str, tuple[int, int]] = defaultdict(lambda: (default_rank, default_points))

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()

        for idx, row in X.iterrows():
            player1 = row["player1"]
            player2 = row["player2"]

            if pd.isna(row["player1_Rank"]):
                X.loc[idx, "player1_Rank"] = self.players[player1][0]
            if pd.isna(row["player2_Rank"]):
                X.loc[idx, "player2_Rank"] = self.players[player2][0]

            if pd.isna(row["player1_Pts"]):
                X.loc[idx, "player1_Pts"] = self.players[player1][1]
            if pd.isna(row["player2_Pts"]):
                X.loc[idx, "player2_Pts"] = self.players[player2][1]

            self.players[player1] = (X.loc[idx, "player1_Rank"], X.loc[idx, "player1_Pts"])
            self.players[player2] = (X.loc[idx, "player2_Rank"], X.loc[idx, "player2_Pts"])

        return X


class BestOfImputer(BaseEstimator, TransformerMixin):
    """Custom transformer to impute missing values in the 'Best of' column"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        missing_best_of : pd.Series = X["Best of"].isna()

        # player1 3-0
        player1_swept : pd.Series = (
            (X["player1_1"] > X["player2_1"]) &
            (X["player1_2"] > X["player2_2"]) &
            (X["player1_3"] > X["player2_3"])
        )

        # player2 3-0
        player2_swept : pd.Series = (
            (X["player1_1"] < X["player2_1"]) &
            (X["player1_2"] < X["player2_2"]) &
            (X["player1_3"] < X["player2_3"])
        )

        # either player1 or player2 swept the match or the match went to at least 4 sets
        best_of_5 : pd.Series = (
            (X["player1_4"] != 0) | (X["player2_4"] != 0) |
            player1_swept | player2_swept
        )

        X.loc[missing_best_of , "Best of"] = np.where(
            best_of_5.loc[missing_best_of],
            5,
            3
        )

        return X
    