import pandas as pd
import numpy as np
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
    

class AlignTypesTransformer(BaseEstimator, TransformerMixin):
    """Align types of the columns"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X : pd.DataFrame):
        cols = [
            col for col in X.select_dtypes(include='float').columns 
            if X[col].dropna().apply(float.is_integer).all() 
        ]

        # game columns are considered as object
        # for col in GAME_COLS:
        #     if col in X.columns and col not in cols:
        #         X[col] = pd.to_numeric(X[col], errors='coerce')
        #         cols.append(col)

        X[cols] = X[cols].astype("Int64")
        return X
    

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
    

# class OutliersTransformer(BaseEstimator, TransformerMixin):
#     """Custom transformer to handle outliers in the dataset"""

#     def __init__(self, n_estimators = 50):
#         self.rf = RandomForestClassifier(
#             random_state=42,
#             n_estimators=n_estimators,
#         )

#     def fit(self, X, y):
#         self.is_fitted_ = True
#         self.rf.fit(X=X, y=y)
#         return self
    
#     def transform(self, X):
#         leaves = self.rf.apply(X)
#         pair_wise_sim = pairwise_distances(
#             X=leaves,
#             metric=lambda i, j : np.mean(i != j)
#         )
#         out = (pair_wise_sim**2).sum(axis=0)**-1
        
#         return X