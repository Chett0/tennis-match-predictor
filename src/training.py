import logging

logger = logging.getLogger(__name__)

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from data_loader import PROCESSED_DATA_PATH, train_test_split


def train_model(df : pd.DataFrame):
    """Train a model on the given dataframe."""

    X_train, X_test, y_train, y_test, years = train_test_split(df)

    tscv = TimeSeriesSplit(n_splits=years - 1)
    model = RandomForestClassifier(random_state=42)

    # fitted_model = RandomizedSearchCV(
    #     estimator=model,
    #     param_distributions={
    #         'n_estimators': [100, 200, 300],
    #         'max_depth': [None, 10, 20, 30],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4],
    #     },
    #     n_iter=10,
    #     cv=tscv,
    #     verbose=2,
    #     scoring='accuracy',
    #     random_state=42,
    # ) 

    fitted_model = RFECV(
        estimator=model,
        step=1,
        cv=tscv,
        scoring='accuracy',
        verbose=2
    )

    fitted_model.fit(X_train, y_train)

    print(f"Model accuracy: {fitted_model.score(X_test, y_test):.2f}")

    return fitted_model

def read_processed_data() -> pd.DataFrame:
    return pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')

def training_pipeling(df : pd.DataFrame):

    logger.info("Starting training pipeline")

    model = train_model(df)
    
    logger.info("Training pipeline completed")

if __name__ == "__main__":
    df = read_processed_data()
    model = train_model(df)
    
    