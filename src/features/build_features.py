import pandas as pd

from src.config import BILL_COLS, DELAY_COLS, PAY_COLS


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AVG_BILL_AMT"] = df[BILL_COLS].mean(axis=1)
    df["AVG_PAY_AMT"] = df[PAY_COLS].mean(axis=1)
    df["TOTAL_BILL_AMT"] = df[BILL_COLS].sum(axis=1)
    df["TOTAL_PAY_AMT"] = df[PAY_COLS].sum(axis=1)
    df["UTILIZATION"] = df[BILL_COLS].div(df["LIMIT_BAL"].replace(0, 1), axis=0).mean(axis=1)
    df["PAY_RATIO"] = df[PAY_COLS].div(df[BILL_COLS].abs() + 1, axis=0).mean(axis=1)
    df["NUM_DELAYED"] = (df[DELAY_COLS] > 0).sum(axis=1)
    df["MAX_DELAY"] = df[DELAY_COLS].max(axis=1)
    df["AGE_BIN"] = (
        pd.cut(df["AGE"], bins=[0, 25, 35, 45, 55, 65, 100], labels=[0, 1, 2, 3, 4, 5])
        .astype(int)
    )
    return df
