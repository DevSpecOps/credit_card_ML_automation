import json
from pathlib import Path

import pandas as pd

from src import config
from src.api.storage import fetch_predictions
from src.features.build_features import engineer_features
from src.monitoring.drift import compute_drift_report


def load_reference() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_PATH)


def load_production_features(limit: int = 2000) -> pd.DataFrame:
    rows = fetch_predictions(limit=limit)
    if not rows:
        return pd.DataFrame()
    features = [json.loads(r[1]) for r in rows]
    return engineer_features(pd.DataFrame(features))


def run(drift_threshold: float = 0.25) -> pd.DataFrame:
    log_path = Path(config.ARTIFACTS_DIR) / "drift_alerts.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    reference = load_reference()
    current = load_production_features()

    if current.empty:
        print("no production predictions to compare")
        log_path.unlink(missing_ok=True)
        return pd.DataFrame()

    report = compute_drift_report(reference, current, config.NUMERIC_FEATURES)
    print(report.to_string(index=False))

    alerts = report[report["psi"] >= drift_threshold]
    if not alerts.empty:
        lines = [
            f"{row['feature']}\t{row['psi']:.4f}\t{row['status']}" for _, row in alerts.iterrows()
        ]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"ALERT: {len(alerts)} features above {drift_threshold} -> {log_path}")
    else:
        log_path.unlink(missing_ok=True)
        print(f"no drift alerts (threshold {drift_threshold})")
    return report


if __name__ == "__main__":
    run()
