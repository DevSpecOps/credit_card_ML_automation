import json
import subprocess
import time
from pathlib import Path

from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

from src import config

ALERTS_PATH = Path(config.ARTIFACTS_DIR) / "drift_alerts.log"
REPORT_PATH = Path(config.ARTIFACTS_DIR) / "retrain_report.json"
DRIFT_THRESHOLD = 0.25
REGISTERED_MODEL_NAME = "credit_card_default_model"
ALIAS_NAME = "staging"


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

    max_retries = 15
    retry_delay = 2
    for attempt in range(max_retries):

        current_status = ModelVersionStatus.from_string(latest.status)
        if current_status == ModelVersionStatus.READY:
            break
        print(
            f"Waiting for model version {latest.version} to be READY... (attempt {attempt+1}/{max_retries})"
        )
        time.sleep(retry_delay)
        latest = client.get_model_version(REGISTERED_MODEL_NAME, latest.version)

    if ModelVersionStatus.from_string(latest.status) != ModelVersionStatus.READY:
        raise RuntimeError(
            f"Model version {latest.version} is not ready after {max_retries * retry_delay}s (status: {latest.status})"
        )

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=ALIAS_NAME,
        version=latest.version,
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
        print(f"new alias '{ALIAS_NAME}' set for version: {version}")
    elif not triggered:
        print("no retrain needed (no alerts above threshold)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
