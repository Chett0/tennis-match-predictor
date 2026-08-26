import logging

import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import pandas as pd

from src.data.loader import load_data, train_test_split
from src.data.preprocess import preprocessing_pipeline, create_labels, filter_rows, DropFeatureSelector
from src.features.features import feature_engineering_pipeline
from src.training.training import fine_tune, training_pipeline
from src.evaluation.metrics import cross_validation, evaluation_metrics, print_metrics
from src.evaluation.plots import plot_confusion_matrix, plot_roc_curve
from src.utils.utils import get_df_from_pipeline

def run_pipeline():
    """Run the entire machine learning pipeline"""
    logger.info("Starting pipeline execution")

    df : pd.DataFrame = load_data()
    df = df[df["WRank"] <= 100]
    create_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(df)
    years = X_train["Date"].dt.year.max() - X_train["Date"].dt.year.min() + 1

    prep_pipeline : Pipeline = preprocessing_pipeline(X_train, y_train)
    feat_pipeline : Pipeline = feature_engineering_pipeline()
    train_pipeline : Pipeline = training_pipeline(
        model=xgb.XGBClassifier()
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", prep_pipeline),
            ("feature_engineering", feat_pipeline),
            # ("drop_date", DropFeatureSelector(features_to_drop=["date"])),
            ("train", train_pipeline)
        ]
    )

    hyperparams = {
        'train__model__n_estimators': [100, 300, 500],
        'train__model__max_depth': [3, 6, 10],
        'train__model__learning_rate': [0.01, 0.05, 0.1],
        'train__model__subsample': [0.6, 0.8, 1.0],
        'train__model__colsample_bytree': [0.6, 0.8, 1.0],
        'train__model__min_child_weight': [1, 3, 5],
        'train__model__gamma': [0, 0.1, 0.3],
        'train__model__reg_lambda': [1, 5, 10]
    }

    fitted_pipeline : RandomizedSearchCV | GridSearchCV = fine_tune(
        pipeline=pipeline,
        tuning_methods=RandomizedSearchCV,
        years=years,
        hyperparams=hyperparams,
        scoring="accuracy",
        random_state=42
    )

    fitted_pipeline.fit(X_train, y_train)

    cross_val_scores = cross_validation(
        pipeline=fitted_pipeline,
        X=X_train,
        y=y_train,
        years=years,
        scoring="accuracy"
    )

    print(pd.Series(cross_val_scores).describe())

    y_pred = fitted_pipeline.predict(X_test)

    metrics = evaluation_metrics(
        model=fitted_pipeline,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred
    )

    print_metrics(metrics)

    plot_confusion_matrix(cm=metrics["confusion_matrix"])
    plot_roc_curve(
        model=fitted_pipeline,
        X_test=X_test,
        y_test=y_test
    )


if __name__ == "__main__":
    run_pipeline()