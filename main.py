import logging

from sklearn.pipeline import Pipeline

logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import pandas as pd

from src.data.loader import load_data
from src.data.preprocess import preprocessing_pipeline
from src.features.features import feature_engineering_pipeline
from src.training.training import training_testing_pipeline
from src.utils.utils import get_df_from_pipeline

def run_pipeline():

    logger.info("Starting pipeline execution")

    df : pd.DataFrame = load_data()

    prep_pipeline : Pipeline = preprocessing_pipeline(df)

    feat_pipeline : Pipeline = feature_engineering_pipeline(df)

    pipeline = Pipeline(
        steps=[
            ("preprocessing", prep_pipeline),
            ("feature_engineering", feat_pipeline)
        ]
    )

    df = get_df_from_pipeline(pipeline, df)
    df.info()

    training_testing_pipeline(df)

if __name__ == "__main__":
    run_pipeline()