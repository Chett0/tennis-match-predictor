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