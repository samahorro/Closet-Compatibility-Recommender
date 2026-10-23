"""Sam: evaluation helpers. Fill-in-the-blank evaluation to be added."""
from sklearn.metrics import accuracy_score, roc_auc_score


def binary_metrics(y_true, y_pred, y_score):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "roc_auc": float(roc_auc_score(y_true, y_score))}
