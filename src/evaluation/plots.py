import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

def plot_roc_curve(
        model, 
        X_test : pd.DataFrame,
        y_test : pd.Series
):
    
    RocCurveDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.show()


def plot_confusion_matrix(
        cm: np.ndarray
):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    plt.title('Confusion Matrix')
    plt.show()


def plot_feature_importance(
        feature_names : list[str],
        feature_importances : pd.Series,
        top_n : int = 20
):
    n = min(top_n, len(feature_importances))
    fig, ax = plt.subplots(figsize=(15,15))
    ax.barh(range(0,n), feature_importances[:n])
    ax.set_title("Feature Importances")
    ax.set_yticks(range(n))
    ax.set_yticklabels(feature_names[:n])
    ax.grid()


def plot_difference(
        dataframe: pd.DataFrame,
        first_col: str,
        second_col: str
):
    """Plot the row-wise difference between two numeric dataframe columns."""
    missing_columns = [
        column for column in (first_col, second_col) if column not in dataframe
    ]
    if missing_columns:
        raise KeyError(f"Columns not found in dataframe: {missing_columns}")

    difference = dataframe[first_col] - dataframe[second_col]

    fig, ax = plt.subplots()
    ax.plot(difference)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Difference between {first_col} and {second_col}")
    ax.set_xlabel("Row")
    ax.set_ylabel(f"{first_col} - {second_col}")
    ax.grid()

    plt.show()


def plot_top100(df : pd.DataFrame):
    """Plot the relationship between winner and loser ranks for matches where both players are in the top 100."""

    # top_100 = df[(df["Date"] >= "2024-01-01") &(df["WRank"] <= 100) & (df["LRank"] <= 100)]
    top_100 = df[(df["WRank"] <= 100) & (df["LRank"] <= 100)]

    plt.scatter(top_100["WRank"], top_100["LRank"], alpha=0.1)
    plt.plot([1, 100], [1, 100], color='red', linestyle='--')
    plt.xlabel("Winner Rank (Top 100)")
    plt.ylabel("Loser Rank (Top 100)")
    plt.title("Winner vs Loser Rankings")

    plt.show()



def plot_rank_difference(df : pd.DataFrame):
    """Plot the distribution of ranking differences between winners and losers."""

    plt.hist(df["WRank"], bins=100, range=(1, 200), alpha=0.5, label='WRank')
    plt.hist(df["LRank"], bins=100, range=(1, 200), alpha=0.5, label='LRank')

    plt.xlabel('Ranking')
    plt.ylabel('Frequency')
    plt.title('Winner - Loser Ranking')
    plt.legend()

    plt.show()