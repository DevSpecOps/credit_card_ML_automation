import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from src import config

DB_PATH = Path(
    os.environ.get(
        "MONITORING_DB_PATH",
        config.PROJECT_ROOT / "monitoring_data" / "monitoring.db",
    )
)


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            features TEXT NOT NULL,
            probability REAL NOT NULL,
            prediction INTEGER NOT NULL
        )
        """)
    return conn


def log_prediction(features: dict, probability: float, prediction: int) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO predictions (ts, features, probability, prediction) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                json.dumps(features, sort_keys=True),
                float(probability),
                int(prediction),
            ),
        )


def fetch_predictions(limit: int = 1000):
    with closing(_connect()) as conn:
        return conn.execute(
            "SELECT ts, features, probability, prediction FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
