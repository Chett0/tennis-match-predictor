import argparse
import logging

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from src.utils.utils import get_df_from_pipeline
from src.data.loader import PROCESSED_DATA_PATH
from src.features.transformers import FeatureEngineringTransformer


def feature_engineering_pipeline() -> Pipeline:

    return Pipeline(
        steps=[
            ("create_features", FeatureEngineringTransformer())
        ]
    )

def save_features(df : pd.DataFrame):
    df.to_excel(f'{PROCESSED_DATA_PATH}\\tennis_matches_features.xlsx', index=False)

def read_clean_data() -> pd.DataFrame:
    return pd.read_excel(f'{PROCESSED_DATA_PATH}\\tennis_matches_clean.xlsx')

def create_features(save_data : bool = False) -> pd.DataFrame:
    
    logger.info("Starting feature engineering pipeline")

    df : pd.DataFrame = read_clean_data()
    pipeline : Pipeline = feature_engineering_pipeline()
    df = get_df_from_pipeline(pipeline, df)
    if save_data:
        save_features(df)

    logger.info("Feature engineering pipeline completed")

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save preprocessed data to Excel file"
    )

    args = parser.parse_args()

    create_features(save_data=args.save)
