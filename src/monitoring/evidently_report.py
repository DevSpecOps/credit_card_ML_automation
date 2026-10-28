import json
from pathlib import Path

import pandas as pd

from src import config
from src.api.storage import fetch_predictions


def load_reference() -> pd.DataFrame:
    return pd.read_csv(config.TRAIN_PATH)


def load_production(limit: int = 2000) -> pd.DataFrame:
    rows = fetch_predictions(limit=limit)
    if not rows:
        return pd.DataFrame()
    features = [json.loads(r[1]) for r in rows]
    return pd.DataFrame(features)


def run() -> None:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = load_reference()
    current = load_production()
    if current.empty:
        print("no production predictions to compare")
        return

    common = [c for c in config.NUMERIC_FEATURES if c in reference.columns and c in current.columns]
    if not common:
        print("no common numeric features between train and production logs")
        return

    reference = reference[common]
    current = current[common]
    print(f"comparing {len(common)} numeric features")

    out_dir = Path(config.ARTIFACTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = Report([DataDriftPreset()])
    snapshot = report.run(reference_data=reference, current_data=current)
    snapshot.save_html(str(out_dir / "drift_report.html"))
    with open(out_dir / "drift_report.json", "w", encoding="utf-8") as f:
        json.dump(snapshot.dict(), f, indent=2, default=str)
    print(f"evidently report -> {out_dir / 'drift_report.html'}")


if __name__ == "__main__":
    run()
