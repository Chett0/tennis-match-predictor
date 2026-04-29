import logging

logging.basicConfig(
    filename='../logs/pipeline.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

import pandas as pd
from src.data import load_data

from preprocess import preprocess_pipeline
from features import feature_engineering_pipeline
from training import training_pipeling

def run_pipeline():

    df : pd.DataFrame = load_data()

    preprocess_pipeline(df)

    feature_engineering_pipeline(df)

    training_pipeling(df)

if __name__ == "__main__":
    run_pipeline()