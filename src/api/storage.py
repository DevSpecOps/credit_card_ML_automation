import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src import config

DB_PATH = Path(config.PROJECT_ROOT) / "monitoring.db"


def _connect():
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
    with _connect() as conn:
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
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, features, probability, prediction FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows
