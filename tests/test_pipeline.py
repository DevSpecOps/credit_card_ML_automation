from src.models.pipeline import build_pipeline


def test_pipeline_fit_predict(raw_df_with_target):
    df = raw_df_with_target.copy()
    y = df.pop("default_payment_next_month")
    pipe = build_pipeline(n_estimators=10, max_depth=2)
    pipe.fit(df, y)
    proba = pipe.predict_proba(df)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba.shape == (len(df),)
