import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import pandas as pd

from src.data.loader import load_data, train_test_split
from src.data.preprocess import preprocessing_pipeline, create_labels, DropFeatureSelector
from src.features.features import feature_engineering_pipeline
from src.training.training import training_testing_pipeline
from src.utils.utils import get_df_from_pipeline

def run_pipeline():

    logger.info("Starting pipeline execution")

    df : pd.DataFrame = load_data()

    create_labels(df)

    X_train, X_test, y_train, y_test = train_test_split(df)

    prep_pipeline : Pipeline = preprocessing_pipeline(X_train)

    feat_pipeline : Pipeline = feature_engineering_pipeline()

    pipeline = Pipeline(
        steps=[
            ("preprocessing", prep_pipeline),
            ("feature_engineering", feat_pipeline),
            ("drop_date", DropFeatureSelector(features_to_drop=["date"])),
            ("RandomForestClassifier", RandomForestClassifier(random_state=42))
        ]
    )

    pipeline.fit(X_train, y_train)

    pipeline.predict(X_test)

    print(pipeline.score(X_test, y_test))

    # pipeline.predict(X_test)

if __name__ == "__main__":
    run_pipeline()