import numpy as np
import pandas as pd

from src import config


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf

    e_cnt, _ = np.histogram(expected, bins=breakpoints)
    a_cnt, _ = np.histogram(actual, bins=breakpoints)

    e_pct = np.where(e_cnt == 0, 1e-6, e_cnt / max(len(expected), 1))
    a_pct = np.where(a_cnt == 0, 1e-6, a_cnt / max(len(actual), 1))
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def compute_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, features: list
) -> pd.DataFrame:
    rows = []
    for f in features:
        if f not in reference.columns or f not in current.columns:
            continue
        value = psi(reference[f].values, current[f].values)
        status = "OK" if value < 0.1 else "WARN" if value < 0.25 else "ALERT"
        rows.append({"feature": f, "psi": value, "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


if __name__ == "__main__":
    ref = pd.read_csv(config.TRAIN_PATH)
    cur = pd.read_csv(config.TEST_PATH)
    report = compute_drift_report(ref, cur, config.NUMERIC_FEATURES)
    print(report.to_string(index=False))
