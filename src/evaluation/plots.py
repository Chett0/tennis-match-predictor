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
    fig, ax = plt.subplots(figsize=(15,15))
    ax.barh(range(0,top_n), feature_importances[:top_n])
    ax.set_title("Feature Importances")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_names[:top_n])
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