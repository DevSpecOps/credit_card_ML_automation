import numpy as np
from src.models.metrics import compute_metrics


def test_compute_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0
