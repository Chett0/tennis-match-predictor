import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils import PROCESSED_DATA_PATH


def train_model(df : pd.DataFrame):
    """Train a model on the given dataframe."""

    years = df['date'].dt.year.max() - df['date'].dt.year.min()

    train_df = df[df["date"] < "2025-01-01"]
    test_df = df[df["date"] >= "2025-01-01"]

    train_df = train_df.drop(columns=["date"])
    test_df = test_df.drop(columns=["date"])

    X_train = train_df.drop(columns=["winner"])
    y_train = train_df["winner"]

    X_test = test_df.drop(columns=["winner"])
    y_test = test_df["winner"]


    tscv = TimeSeriesSplit(n_splits=years - 1)
    model = RandomForestClassifier(random_state=42)

    fitted_model = RandomizedSearchCV(
        estimator=model,
        param_distributions={
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        n_iter=10,
        cv=tscv,
        verbose=2,
        scoring='accuracy',
        random_state=42,
    ) 

    fitted_model.fit(X_train, y_train)

    print(f"Model accuracy: {fitted_model.score(X_test, y_test):.2f}")

    return fitted_model

def training_pipeling():
    df = pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')
    model = train_model(df)

if __name__ == "__main__":
    df = pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')
    model = train_model(df)
    
    