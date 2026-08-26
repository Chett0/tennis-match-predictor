from typing import cast

import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score
)
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV, 
    TimeSeriesSplit, 
    cross_val_score
)
from sklearn.pipeline import Pipeline


def evaluation_metrics(
        model, 
        X_test : pd.DataFrame, 
        y_test : pd.Series,
        y_pred : np.ndarray
) -> dict:

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics


def cross_validation(
        pipeline :  RandomizedSearchCV | GridSearchCV,
        X : pd.DataFrame,
        y : pd.Series,
        years : int,
        verbose : int = 0,
        scoring : str = "accuracy"
) -> np.ndarray:
    
    tscv = TimeSeriesSplit(n_splits=years - 1)
    scores = cross_val_score(
        estimator=pipeline,
        X=X, 
        y=y,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1,
        verbose=verbose
    )

    return scores


def feature_importance(
        pipeline : RandomizedSearchCV | GridSearchCV,
        X : pd.DataFrame,
        model_step : str = "train",
) -> pd.DataFrame:

    best = cast(Pipeline, pipeline.best_estimator_)
    model = best.named_steps[model_step].named_steps["model"]

    transformers = best[:-1]
    names = transformers.transform(X).columns

    importances = pd.DataFrame(
        {"importance": model.feature_importances_},
        index=names
    )

    importances.sort_values(by="importance", ascending=False, inplace=True)
    return importances


def print_metrics(metrics : dict) -> None:
    """Pretty-print the evaluation metrics dictionary.

    Args:
        metrics: mapping from metric name to its value, as returned by ``evaluation_metrics``.
    """
    for name, value in metrics.items():
        print(f"\n{name.replace('_', ' ').title()}:")

        if name == "confusion_matrix":
            print(pd.DataFrame(value))
        elif isinstance(value, (float, np.floating)):
            print(f"{value:.4f}")
        else:
            print(value)
