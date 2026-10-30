import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import (
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV

from src import config
from src.features.build_features import engineer_features
from src.models.pipeline import build_pipeline

REGISTERED_MODEL_NAME = "credit_card_default_model"


def load_params() -> dict:
    path = config.PROJECT_ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_split(path: Path):
    df = pd.read_csv(path)
    if "AVG_BILL_AMT" not in df.columns or "AGE_BIN" not in df.columns:
        df = engineer_features(df)
    y = df[config.TARGET].astype(int)
    X = df.drop(columns=[config.TARGET])
    return X, y


def train() -> None:
    start_time = time.time()
    print("[1/6] Loading parameters...", flush=True)
    params = load_params()["train"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI))
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    print("[2/6] Loading train/test data...", flush=True)
    X_train, y_train = load_split(config.TRAIN_PATH)
    X_test, y_test = load_split(config.TEST_PATH)
    print(f"      Train shape: {X_train.shape}, Test shape: {X_test.shape}", flush=True)

    param_grid = {
        "clf__n_estimators": params["n_estimators"],
        "clf__max_depth": params["max_depth"],
        "clf__learning_rate": params["learning_rate"],
    }
    total_fits = (
        len(params["n_estimators"])
        * len(params["max_depth"])
        * len(params["learning_rate"])
        * params["cv"]
    )
    print(f"[3/6] Starting GridSearchCV ({total_fits} fits)...", flush=True)

    with mlflow.start_run() as run:
        grid = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=param_grid,
            cv=params["cv"],
            scoring=params["scoring"],
            n_jobs=-1,
            verbose=2,
        )
        grid.fit(X_train, y_train)

        print(f"[4/6] GridSearch completed in {time.time() - start_time:.1f}s.", flush=True)
        print(f"      Best params: {grid.best_params_}", flush=True)

        best = grid.best_estimator_
        mlflow.log_params(grid.best_params_)

        print("[5/6] Calculating metrics and logging to MLflow...", flush=True)
        y_proba = best.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)

        Path(config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name="Model")
        ax.set_title(f"ROC Curve (AUC = {metrics['roc_auc']:.4f})")

        roc_path = Path(config.ARTIFACTS_DIR) / "roc_curve.png"
        fig.savefig(roc_path, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifact(str(roc_path), artifact_path="plots")

        print("[6/6] Saving model and artifacts...", flush=True)
        Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(best, config.MODEL_PATH)

        mlflow.sklearn.log_model(
            best,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
                "sklearn.compose._column_transformer._RemainderColsList",
            ],
        )

        with open(Path(config.ARTIFACTS_DIR) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": run.info.run_id, **metrics}, f, indent=2)

        print("=" * 40, flush=True)
        print(f"run_id = {run.info.run_id}")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}")
        print(f"Total time: {time.time() - start_time:.1f}s", flush=True)
        print("=" * 40, flush=True)


if __name__ == "__main__":
    train()
