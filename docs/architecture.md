# Architecture

## Layers

```mermaid
flowchart LR
    A[data/raw] -->|dvc prepare| B[data/processed]
    B -->|dvc train| C[models/model.pkl]
    C -->|Docker volume| D[FastAPI service]
    D -->|log| E[(monitoring.db)]
    E -->|PSI + Evidently| F[drift alerts]
    F -->|threshold| G[retrain trigger]
    G -->|dvc repro --force| C
    C -.->|register| H[MLflow Registry]
```

## Reproduce the whole pipeline

    dvc repro
    docker compose up -d
    python -m src.monitoring.simulate_traffic 100
    python -m src.monitoring.report
    python -m src.monitoring.evidently_report
