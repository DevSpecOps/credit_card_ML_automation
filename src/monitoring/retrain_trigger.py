import json
import subprocess
from pathlib import Path

from mlflow.tracking import MlflowClient

from src import config

ALERTS_PATH = Path(config.ARTIFACTS_DIR) / "drift_alerts.log"
REPORT_PATH = Path(config.ARTIFACTS_DIR) / "retrain_report.json"
DRIFT_THRESHOLD = 0.25
REGISTERED_MODEL_NAME = "credit_card_default_model"


def read_alerts() -> list:
    if not ALERTS_PATH.exists():
        return []
    rows = []
    for line in ALERTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            rows.append({"feature": parts[0], "psi": float(parts[1])})
        except ValueError:
            continue
    return rows


def should_retrain(alerts: list, threshold: float = DRIFT_THRESHOLD) -> bool:
    return any(a["psi"] >= threshold for a in alerts)


def promote_latest(client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return "no versions"
    latest = max(versions, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        REGISTERED_MODEL_NAME,
        latest.version,
        stage="Staging",
        archive_existing_versions=False,
    )
    return latest.version


def run(dry_run: bool = False) -> dict:
    alerts = read_alerts()
    triggered = should_retrain(alerts)
    report = {"alerts": alerts, "triggered": triggered}

    if triggered and not dry_run:
        cmd = ["dvc", "repro", "--force"]
        print("running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
        client = MlflowClient(config.MLFLOW_TRACKING_URI)
        version = promote_latest(client)
        report["new_version"] = version
        print(f"new Staging version: {version}")
    elif not triggered:
        print("no retrain needed (no alerts above threshold)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
