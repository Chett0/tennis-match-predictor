import logging
from typing import Type

from sklearn.inspection import permutation_importance

from evaluation import evaluate_model

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_loader import PROCESSED_DATA_PATH, train_test_split


def train_test(
        df : pd.DataFrame,
        model_class : Type[RandomForestClassifier] | Type[XGBClassifier],
        tuning_methods : list[Type[BaseSearchCV]],
        hyperparams : dict = {},
        test_year : int | None = None,
        drop_columns : list = [],
        random_state : int = 42
    ) -> BaseEstimator:
    """Train a model on the given dataframe."""

    df.drop(
        columns=drop_columns, 
        inplace=True, 
        errors="ignore"
    )

    if not test_year:
        X_train, X_test, y_train, y_test, years = train_test_split(df)
    else:
        X_train, X_test, y_train, y_test, years = train_test_split(df, test_year=test_year)

    tscv = TimeSeriesSplit(n_splits=years - 1)
    
    
    if model_class == RandomForestClassifier:
        model = RandomForestClassifier(random_state=random_state)
    elif model_class == XGBClassifier:
        model = XGBClassifier(random_state=random_state)
    else:
        raise ValueError("Unsupported model class")
    
    if RandomizedSearchCV in tuning_methods:
        fitted_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=hyperparams,
            n_iter=10,
            cv=tscv,
            verbose=2,
            scoring='accuracy',
            random_state=random_state,
            n_jobs=-1
        )
    elif GridSearchCV in tuning_methods:
        fitted_model = GridSearchCV(
            estimator=model,
            param_grid=hyperparams,
            cv=tscv,
            verbose=2,
            scoring='accuracy',
            n_jobs=-1
        )
    else:
        raise ValueError("Unsupported tuning method")

    fitted_model.fit(X_train, y_train)
    best_model = fitted_model.best_estimator_
    evaluate_model(best_model, X_test, y_test)
    
    return best_model



def read_processed_data() -> pd.DataFrame:
    return pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')

def training_testing_pipeline(df : pd.DataFrame):

    logger.info("Starting training pipeline")

    hyperparams = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    model = train_test(
        df,
        model_class=RandomForestClassifier,
        tuning_methods=[RandomizedSearchCV],
        hyperparams=hyperparams
    )
    
    logger.info("Training pipeline completed")

    return model

if __name__ == "__main__":
    df = read_processed_data()
    model = training_testing_pipeline(df)
    
    