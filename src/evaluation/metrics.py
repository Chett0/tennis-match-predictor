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
        X_test : pd.DataFrame | np.ndarray, 
        y_test : pd.Series,
        y_pred
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
        pipeline :  RandomizedSearchCV | GridSearchCV | Pipeline,
        X : pd.DataFrame | np.ndarray,
        y : pd.Series,
        tscv : TimeSeriesSplit = TimeSeriesSplit(n_splits=5),
        verbose : int = 0,
        scoring : str = "accuracy"
) -> np.ndarray:
    
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
        pipeline : Pipeline,
        X : pd.DataFrame,
        model_step : str = "training",
        coef : bool = False
) -> pd.DataFrame:

    model = pipeline.named_steps[model_step]

    transformers = pipeline[:-1]
    names = transformers.transform(X).columns

    if not coef:
        importances = pd.DataFrame(
            {"importance": model.feature_importances_},
            index=names
        )
    else:
        coefs = pipeline.named_steps['training'].coef_[0]
        importances = pd.DataFrame(
            {"importance": coefs},
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
