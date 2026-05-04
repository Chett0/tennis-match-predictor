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

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline


def evaluation_metrics(
        model, 
        X_test : pd.DataFrame, 
        y_test : pd.Series
) -> dict:

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
        "f1": f1_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    return metrics


def cross_validation(
        pipeline : Pipeline | RandomizedSearchCV | GridSearchCV,
        X : pd.DataFrame,
        y : pd.Series,
        years : int,
        scoring : str = "accuracy"
) -> np.ndarray:
    
    tscv = TimeSeriesSplit(n_splits=years - 1)
    scores = cross_val_score(
        estimator=pipeline,
        X=X, 
        y=y,
        cv=tscv,
        scoring=scoring,
        n_jobs=-1
    )

    return scores


def feature_importance(
        pipeline : RandomizedSearchCV | GridSearchCV,
        pipeline_step_classifier_name : str,
        pipeline_step_features_name : str,
):
    best_estimator = cast(Pipeline, pipeline.best_estimator_)
    feature_importance = best_estimator.named_steps[
        pipeline_step_classifier_name
    ].feature_importances_

    feature_names = best_estimator.named_steps[
        pipeline_step_features_name
    ].get_feature_names_out()


    return sorted(
        zip(feature_importance, feature_names),
        reverse=True
    )