from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import config
from src.api.storage import log_prediction
from src.features.build_features import engineer_features

app = FastAPI(title="credit_card_default")

_model = None


def get_model():
    global _model
    if _model is None:
        if not Path(config.MODEL_PATH).exists():
            raise HTTPException(status_code=503, detail="model not trained")
        _model = joblib.load(config.MODEL_PATH)
    return _model


class ClientData(BaseModel):
    LIMIT_BAL: float = Field(..., ge=0)
    AGE: int = Field(..., ge=18, le=120)
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(client: ClientData) -> dict:
    data = client.model_dump() if hasattr(client, "model_dump") else client.dict()
    df = pd.DataFrame([data])
    df = engineer_features(df)
    model = get_model()
    proba = float(model.predict_proba(df)[:, 1][0])
    log_prediction(data, proba, int(proba >= 0.5))
    return {"default_prediction": int(proba >= 0.5), "default_probability": proba}
