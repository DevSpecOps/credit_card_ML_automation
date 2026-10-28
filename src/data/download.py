import urllib.request
from pathlib import Path

import pandas as pd

from src import config

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"


def download_dataset() -> None:
    if config.DATA_RAW.exists():
        print(f"Dataset already exists at {config.DATA_RAW}")
        return

    config.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    temp_xls = config.DATA_RAW.with_suffix(".xls")

    print("Downloading original XLS from UCI...")
    try:
        urllib.request.urlretrieve(UCI_URL, temp_xls)
        print("XLS downloaded. Converting to CSV...")

        # header=1 пропускает первую строку с ID
        df = pd.read_excel(temp_xls, header=1)
        df = df.rename(columns={"default payment next month": "default.payment.next.month"})
        df.to_csv(config.DATA_RAW, index=False)

        temp_xls.unlink()
        print(f"Dataset saved to {config.DATA_RAW}")
    except Exception as e:
        print(f"Failed to download or convert dataset: {e}")
        if temp_xls.exists():
            temp_xls.unlink()
        raise RuntimeError("Dataset download failed.") from e


if __name__ == "__main__":
    download_dataset()
