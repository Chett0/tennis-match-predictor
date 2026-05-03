import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropFeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to drop specified features"""

    def __init__(self, features_to_drop: list[str]):
        self.features_to_drop = features_to_drop

    def fit(self, X, y=None):
        self.features_to_drop_ = list(self.features_to_drop)
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_drop_, errors="ignore")
    


class AlignTypesTransformer(BaseEstimator, TransformerMixin):
    """Convert selected columns to selected types"""

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

        X[cols] = X[cols].astype("Int64")

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
        missing_best_of = X["Best of"].isna()

        best_of_5 = (
            (X["player1_4"] != 0) | (X["player2_4"] != 0) |
            (
                (X["player1_1"] > X["player2_1"]) &
                (X["player1_2"] > X["player2_2"]) &
                (X["player1_3"] > X["player2_3"])
            )
        )

        X.loc[missing_best_of , "Best of"] = np.where(
            best_of_5.loc[missing_best_of],
            5,
            3
        )

        return X