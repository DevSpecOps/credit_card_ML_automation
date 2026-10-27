from src.features.build_features import engineer_features


def test_no_nan_in_engineered(raw_df):
    out = engineer_features(raw_df.copy())
    key_cols = [
        "AVG_BILL_AMT",
        "AVG_PAY_AMT",
        "TOTAL_BILL_AMT",
        "TOTAL_PAY_AMT",
        "UTILIZATION",
        "PAY_RATIO",
        "NUM_DELAYED",
        "MAX_DELAY",
    ]
    assert out[key_cols].isna().sum().sum() == 0


def test_pay_ratio_finite_on_zero_limit(raw_df):
    df = raw_df.copy()
    df["LIMIT_BAL"] = 0
    out = engineer_features(df)
    assert out["PAY_RATIO"].notna().all()
    assert (out["PAY_RATIO"].abs() < 1e9).all()
