import logging
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
    LIMIT_BAL: float = Field(..., gt=0, le=1_000_000)
    AGE: int = Field(..., ge=18, le=100)
    SEX: int = Field(..., ge=1, le=2)
    EDUCATION: int = Field(..., ge=1, le=4)
    MARRIAGE: int = Field(..., ge=1, le=3)
    PAY_0: int = Field(..., ge=-2, le=9)
    PAY_2: int = Field(..., ge=-2, le=9)
    PAY_3: int = Field(..., ge=-2, le=9)
    PAY_4: int = Field(..., ge=-2, le=9)
    PAY_5: int = Field(..., ge=-2, le=9)
    PAY_6: int = Field(..., ge=-2, le=9)
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float = Field(..., ge=0)
    PAY_AMT2: float = Field(..., ge=0)
    PAY_AMT3: float = Field(..., ge=0)
    PAY_AMT4: float = Field(..., ge=0)
    PAY_AMT5: float = Field(..., ge=0)
    PAY_AMT6: float = Field(..., ge=0)


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
    try:
        log_prediction(data, proba, int(proba >= 0.5))
    except Exception:
        logging.getLogger(__name__).exception("prediction logging failed")
    return {"default_prediction": int(proba >= 0.5), "default_probability": proba}
