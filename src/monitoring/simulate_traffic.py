import json
import random
import sys
from urllib.request import Request, urlopen

import pandas as pd

from src import config


def build_payload(row: dict) -> dict:
    keys = [
        "LIMIT_BAL",
        "AGE",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]
    return {k: row[k] for k in keys}


def simulate(n: int, url: str) -> None:
    df = pd.read_csv(config.TEST_PATH)
    sample = df.sample(n=min(n, len(df)), random_state=random.randint(0, 10**6))
    ok = 0
    for _, row in sample.iterrows():
        payload = json.dumps(build_payload(row.to_dict())).encode()
        req = Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urlopen(req) as resp:
            if resp.status == 200:
                ok += 1
    print(f"sent={n} ok={ok}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000/predict"
    simulate(n, url)
