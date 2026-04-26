import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from utils import PROCESSED_DATA_PATH


def train_model(df : pd.DataFrame) -> RandomForestClassifier:
    """Train a model on the given dataframe."""

    df.drop(columns=["player_0", "player_1"], inplace=True)

    X = df.drop(columns=["winner"])
    y = df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print(f"Model accuracy: {model.score(X_test, y_test):.2f}")

    return model 

if __name__ == "__main__":
    df = pd.read_excel(f'{PROCESSED_DATA_PATH}tennis_matches_features.xlsx')
    print(df.describe().transpose())
    model = train_model(df)
    
    