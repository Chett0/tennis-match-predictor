import logging
from typing import List, Type

from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.base import BaseEstimator

from src.data.loader import PROCESSED_DATA_PATH


def fine_tune(
        pipeline: Pipeline | BaseEstimator,
        tuning_methods : Type[BaseSearchCV],
        hyperparams : dict  | List[dict] ,
        tscv : TimeSeriesSplit = TimeSeriesSplit(n_splits=5),
        scoring : str = "accuracy",
        n_iter : int = 10,
        verbose : int = 2, 
        n_jobs : int = -1,
        random_state : int = 42
    ) -> RandomizedSearchCV | GridSearchCV:
    """Fine tune a pipeline using specified tuning methods and hyperparameters."""
    
    if tuning_methods == RandomizedSearchCV:
        search = RandomizedSearchCV(
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
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=hyperparams,
            cv=tscv,
            verbose=verbose,
            scoring=scoring,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Unsupported tuning method")
    
    return search 


def feature_selection_pipeline(
        model: BaseEstimator,
        tscv: TimeSeriesSplit = TimeSeriesSplit(n_splits=5),
        scoring: str = "accuracy",
        min_features_to_select: int = 1
    ) -> Pipeline:

    return Pipeline(
        steps=[
            ("feature_selection", RFECV(
                estimator=model,
                step=1,
                cv=tscv,
                scoring=scoring,
                min_features_to_select=min_features_to_select,
                n_jobs=-1,
            ))
        ]
    )


def read_processed_data() -> pd.DataFrame:
    return pd.read_excel(PROCESSED_DATA_PATH / "tennis_matches_features.xlsx")


def training_pipeline(
        model : BaseEstimator
) -> Pipeline:
    
    return Pipeline(
        steps=[
            ("training", model)
        ]
    )