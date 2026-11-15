from unittest.mock import patch

import pandas as pd

from src import config
from src.monitoring import report


@patch("src.monitoring.report.compute_drift_report")
@patch("src.monitoring.report.load_production_features")
@patch("src.monitoring.report.load_reference")
def test_run_clears_alerts_when_no_drift(mock_ref, mock_prod, mock_drift, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    log_file = tmp_path / "drift_alerts.log"
    log_file.write_text("AGE\t0.4\tALERT\n", encoding="utf-8")

    mock_ref.return_value = pd.DataFrame()
    mock_prod.return_value = pd.DataFrame({"AGE": [30, 40]})
    mock_drift.return_value = pd.DataFrame([{"feature": "AGE", "psi": 0.05, "status": "OK"}])

    report.run(drift_threshold=0.25)

    assert not log_file.exists()


@patch("src.monitoring.report.load_production_features")
@patch("src.monitoring.report.load_reference")
def test_run_clears_alerts_when_empty_production(mock_ref, mock_prod, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    log_file = tmp_path / "drift_alerts.log"
    log_file.write_text("AGE\t0.4\tALERT\n", encoding="utf-8")

    mock_ref.return_value = pd.DataFrame()
    mock_prod.return_value = pd.DataFrame()

    report.run()

    assert not log_file.exists()
