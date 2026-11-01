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


def get_metric(client: MlflowClient, version: str, metric_name: str = "roc_auc") -> float:
    mv = client.get_model_version(REGISTERED_MODEL_NAME, version)
    run = client.get_run(mv.run_id)
    return float(run.data.metrics.get(metric_name, 0.0))


def promote_if_better(client: MlflowClient) -> str:
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        return "no versions"

    latest = max(versions, key=lambda v: int(v.version))

    max_retries = 15
    for attempt in range(max_retries):
        if ModelVersionStatus.from_string(latest.status) == ModelVersionStatus.READY:
            break
        print(f"Waiting for version {latest.version} to be READY... (attempt {attempt+1})")
        time.sleep(2)
        latest = client.get_model_version(REGISTERED_MODEL_NAME, latest.version)

    if ModelVersionStatus.from_string(latest.status) != ModelVersionStatus.READY:
        raise RuntimeError(f"Version {latest.version} failed to reach READY status.")

    latest_metric = get_metric(client, latest.version)

    try:
        current_staging = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_NAME)
        if current_staging.version == latest.version:
            return f"Version {latest.version} is already {ALIAS_NAME}"

        staging_metric = get_metric(client, current_staging.version)
        print(f"Challenger (v{latest.version}): ROC-AUC = {latest_metric:.4f}")
        print(f"Champion   (v{current_staging.version}): ROC-AUC = {staging_metric:.4f}")

        if latest_metric <= staging_metric:
            print(f"Rejecting v{latest.version}: metric is not better than champion.")
            return f"Version {latest.version} rejected (worse than {current_staging.version})"
    except Exception:
        print(f"No current {ALIAS_NAME} alias found. Promoting v{latest.version} by default.")

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
        version = promote_if_better(client)
        report["promotion_result"] = version
        print(f"Promotion result: {version}")
    elif not triggered:
        print("no retrain needed (no alerts above threshold)")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"report -> {REPORT_PATH}")
    return report


if __name__ == "__main__":
    run()
