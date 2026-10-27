from pathlib import Path

import joblib
import pytest

from src import config
from src.models.pipeline import build_pipeline


@pytest.mark.skipif(not Path(config.MODEL_PATH).exists(), reason="model not trained")
def test_model_is_loadable_and_predicts(raw_df_with_target):
    df = raw_df_with_target.copy()
    df.pop("default_payment_next_month")
    model = joblib.load(config.MODEL_PATH)
    proba = model.predict_proba(df)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_default_model_smoke(raw_df_with_target):
    df = raw_df_with_target.copy()
    y = df.pop("default_payment_next_month")
    pipe = build_pipeline(n_estimators=10, max_depth=2)
    pipe.fit(df, y)
    assert hasattr(pipe, "predict_proba")
