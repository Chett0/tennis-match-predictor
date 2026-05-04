import logging

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
from src.evaluation.evaluation import cross_validation, evaluation_metrics
from src.utils.utils import get_df_from_pipeline

def run_pipeline():

    logger.info("Starting pipeline execution")

    df : pd.DataFrame = load_data()

    create_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(df)
    years = X_train["Date"].dt.year.max() - X_train["Date"].dt.year.min() + 1

    X_train, y_train = filter_rows([
            lambda df: df["Comment"] != "Walkover",
            lambda df: df["player1_sets"].notna() & df["player2_sets"].notna()
        ], X_train, y_train)

    prep_pipeline : Pipeline = preprocessing_pipeline(X_train)

    feat_pipeline : Pipeline = feature_engineering_pipeline()

    train_pipeline : Pipeline = training_pipeline(
        model=RandomForestClassifier(random_state=42)
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", prep_pipeline),
            ("feature_engineering", feat_pipeline),
            ("drop_date", DropFeatureSelector(features_to_drop=["date"])),
            ("train", train_pipeline)
        ]
    )

    hyperparams = {
        'train__model__n_estimators': [100, 200, 300],
        'train__model__max_depth': [None, 10, 20],
        'train__model__min_samples_split': [2, 5, 10],
        'train__model__min_samples_leaf': [1, 2, 4]
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

    print(cross_val_scores)

    metrics = evaluation_metrics(
        model=fitted_pipeline,
        X_test=X_test,
        y_test=y_test
    )

    print(metrics)

if __name__ == "__main__":
    run_pipeline()