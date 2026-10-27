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