import logging

logging.basicConfig(
    filename='../logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import pandas as pd

from data_loader import load_data
from preprocess import preprocess_pipeline
from features import feature_engineering_pipeline
from training import training_pipeline

def run_pipeline():

    logger.info("Starting pipeline execution")

    df : pd.DataFrame = load_data()

    df = preprocess_pipeline(df)

    df = feature_engineering_pipeline(df)

    training_pipeline(df)

if __name__ == "__main__":
    run_pipeline()