import logging
from typing import Type

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator

from src.data.loader import PROCESSED_DATA_PATH, train_test_split


def fine_tune(
        pipeline: Pipeline,
        tuning_methods : Type[BaseSearchCV],
        years : int,
        hyperparams : dict = {},
        scoring : str = "accuracy",
        n_iter : int = 10,
        verbose : int = 2, 
        n_jobs : int = -1,
        random_state : int = 42
    ) -> RandomizedSearchCV | GridSearchCV:
    """Fine tune a pipeline using specified tuning methods and hyperparameters."""

    tscv = TimeSeriesSplit(n_splits=years)
    
    if tuning_methods == RandomizedSearchCV:
        fitted_pipeline = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=hyperparams,
            n_iter=n_iter,
            cv=tscv,
            verbose=verbose,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs
        )
    elif tuning_methods == GridSearchCV:
        fitted_pipeline = GridSearchCV(
            estimator=pipeline,
            param_grid=hyperparams,
            cv=tscv,
            verbose=verbose,
            scoring=scoring,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Unsupported tuning method")
    
    return fitted_pipeline 



def read_processed_data() -> pd.DataFrame:
    return pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')

def training_pipeline(
        model : BaseEstimator
) -> Pipeline:
    
    return Pipeline(
        steps=[
            ("model", model)
        ]
    )

    
    