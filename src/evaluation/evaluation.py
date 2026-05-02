import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    f1_score,
    recall_score,
    accuracy_score
)


def evaluate_model(
        model, 
        X_test : pd.DataFrame, 
        y_test : pd.Series
):

    y_pred = model.predict(X_test)

    print("Model Accuracy:")
    print(f"{accuracy_score(y_test, y_pred):.2f}")
    print("ROC AUC Score:")
    print(f"{roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
    print("F1 Score:")
    print(f"{f1_score(y_test, y_pred):.2f}")
    print("Recall Score:")
    print(f"{recall_score(y_test, y_pred):.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
