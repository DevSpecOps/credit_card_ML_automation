# Credit Default PD - MLOps Pipeline

End-to-end MLOps pipeline for a Probability-of-Default model on the UCI Credit Card Default dataset.

## Stack
Python 3.10, scikit-learn, Pandera, Great Expectations, MLflow, DVC, FastAPI, Docker, GitHub Actions, pytest, flake8, black.

## Structure
src/data data loading and validation
src/features feature engineering
src/models sklearn pipeline and training
src/api FastAPI app
src/monitoring PSI drift and traffic simulator
tests unit and integration tests
notebooks 01_eda.ipynb

## Status
Sprint 1: data layer (download, validation, feature engineering).
## Sprint 3: serving, prediction logging, drift monitoring

- FastAPI service with `/health` and `/predict` (Dockerized).
- Predictions logged to `monitoring.db` (SQLite, git-ignored).
- Traffic simulator: `python -m src.monitoring.simulate_traffic 100`.
- Drift report vs training distribution: `python -m src.monitoring.report` (alerts to `artifacts/drift_alerts.log` when PSI >= 0.25).
- Model registered in MLflow Model Registry as `credit_card_default_model`.

Reproduce:

    dvc repro
    docker compose up -d
    python -m src.monitoring.simulate_traffic 100
    python -m src.monitoring.report
