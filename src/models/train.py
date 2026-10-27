import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src import config
from src.features.build_features import engineer_features
from src.models.pipeline import build_pipeline


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
    params = load_params()["train"]

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.MLFLOW_TRACKING_URI))
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    X_train, y_train = load_split(config.TRAIN_PATH)
    X_test, y_test = load_split(config.TEST_PATH)

    param_grid = {
        "clf__n_estimators": params["n_estimators"],
        "clf__max_depth": params["max_depth"],
        "clf__learning_rate": params["learning_rate"],
    }

    with mlflow.start_run() as run:
        grid = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=param_grid,
            cv=params["cv"],
            scoring=params["scoring"],
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        mlflow.log_params(grid.best_params_)

        y_proba = best.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "f1": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)

        Path(config.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(best, config.MODEL_PATH)
        mlflow.sklearn.log_model(
            best,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

        Path(config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
        with open(Path(config.ARTIFACTS_DIR) / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"run_id": run.info.run_id, **metrics}, f, indent=2)

        print(f"run_id = {run.info.run_id}")
        for k, v in metrics.items():
            print(f"{k:>10}: {v:.4f}")


if __name__ == "__main__":
    train()
