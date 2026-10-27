import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src import config
from src.data.validation import validate_raw
from src.features.build_features import engineer_features


def load_prepare_params() -> dict:
    with open(config.PROJECT_ROOT / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["prepare"]


def prepare(raw_path: Path, out_dir: Path) -> None:
    params = load_prepare_params()

    df = pd.read_csv(raw_path)
    df = df.rename(columns={"default.payment.next.month": config.TARGET})
    validate_raw(df)

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    df = engineer_features(df)

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df[config.TARGET],
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


if __name__ == "__main__":
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_RAW
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else config.DATA_PROCESSED
    prepare(raw, out)
