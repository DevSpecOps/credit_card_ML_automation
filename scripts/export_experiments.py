"""Export all runs of the MLflow experiment to CSV for submission."""
import csv
from pathlib import Path

from mlflow.tracking import MlflowClient

from src import config

OUT = Path("docs/proof/experiments.csv")


def main() -> None:
    client = MlflowClient(config.MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(config.EXPERIMENT_NAME)
    if exp is None:
        raise SystemExit(f"experiment '{config.EXPERIMENT_NAME}' not found")

    runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"])
    if not runs:
        raise SystemExit("no runs found")

    rows = []
    for r in runs:
        m = r.data.metrics
        p = r.data.params
        rows.append(
            {
                "run_id": r.info.run_id,
                "run_name": r.info.run_name or "",
                "status": r.info.status,
                "n_estimators": p.get("clf__n_estimators", ""),
                "max_depth": p.get("clf__max_depth", ""),
                "learning_rate": p.get("clf__learning_rate", ""),
                "roc_auc": f"{m['roc_auc']:.4f}" if "roc_auc" in m else "",
                "f1": f"{m['f1']:.4f}" if "f1" in m else "",
                "precision": f"{m['precision']:.4f}" if "precision" in m else "",
                "recall": f"{m['recall']:.4f}" if "recall" in m else "",
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} runs exported to {OUT}")


if __name__ == "__main__":
    main()