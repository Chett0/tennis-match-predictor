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
    fig, ax = plt.subplots(figsize=(9,4))
    ax.barh(range(0,top_n), feature_importances[:top_n])
    ax.set_title("Feature Importances")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_names[:top_n])
    ax.grid()