import pandas as pd
import pytest

from src import config
from src.features.build_features import engineer_features

collect_ignore_glob = []
if not config.DATA_RAW.exists():
    collect_ignore_glob = ["*validation*.py", "*dataset*.py", "*integration*.py"]


@pytest.fixture
def raw_df():
    data = {
        "ID": [1, 2, 3, 4],  # Добавили ID, который ждет Pandera
        "LIMIT_BAL": [50000, 100000, 30000, 120000],
        "AGE": [30, 45, 22, 51],
        "SEX": [1, 2, 1, 2],
        "EDUCATION": [1, 2, 1, 3],
        "MARRIAGE": [1, 2, 1, 2],
        "default_payment_next_month": [0, 1, 0, 1],
    }
    for col in config.DELAY_COLS:
        data[col] = [0, 2, 0, 1]
    for col in config.BILL_COLS:
        data[col] = [1000, 2000, 300, 4000]
    for col in config.PAY_COLS:
        data[col] = [500, 0, 100, 900]
    return pd.DataFrame(data)


@pytest.fixture
def engineered_df(raw_df):
    return engineer_features(raw_df.copy())


@pytest.fixture
def raw_df_with_target(engineered_df):
    return engineered_df
