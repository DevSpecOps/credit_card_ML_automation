from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_proba, threshold=0.5) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
