# Runbook

## Start API locally

    uvicorn src.api.app:app --port 8000
    curl http://127.0.0.1:8000/health

## Start API in Docker

    docker compose up -d
    curl http://127.0.0.1:8000/health
    docker compose down

## Retrain on drift

    python -m src.monitoring.report
    python -m src.monitoring.retrain_trigger
    python -c "import mlflow; mlflow.set_tracking_uri('sqlite:///mlflow.db'); print(mlflow.search_registered_models())"

## Promote a version to Production

    python -c "from mlflow.tracking import MlflowClient; MlflowClient('sqlite:///mlflow.db').transition_model_version_stage('credit_card_default_model', 2, 'Production')"

## Rollback

    python -c "from mlflow.tracking import MlflowClient; MlflowClient('sqlite:///mlflow.db').transition_model_version_stage('credit_card_default_model', 1, 'Production')"

## Inspect experiments

    mlflow ui --backend-store-uri sqlite:///mlflow.db
    dvc exp show --no-pager
