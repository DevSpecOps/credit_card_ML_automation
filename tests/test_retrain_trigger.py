from src.monitoring.retrain_trigger import should_retrain


def test_no_alerts_no_retrain():
    assert should_retrain([]) is False


def test_small_psi_no_retrain():
    assert should_retrain([{"feature": "AGE", "psi": 0.05}]) is False


def test_large_psi_triggers_retrain():
    assert should_retrain([{"feature": "AGE", "psi": 0.4}]) is True


def test_threshold_is_inclusive():
    assert should_retrain([{"feature": "AGE", "psi": 0.25}], threshold=0.25) is True


def test_log_cleared_when_no_alerts(tmp_path, monkeypatch):
    """After drift disappears, retrain_trigger must not see stale alerts."""
    from src.monitoring import retrain_trigger

    log = tmp_path / "drift_alerts.log"
    log.write_text("AGE\t0.4\tALERT\n", encoding="utf-8")
    monkeypatch.setattr(retrain_trigger, "ALERTS_PATH", log)
    assert retrain_trigger.should_retrain(retrain_trigger.read_alerts()) is True

    log.unlink()
    assert retrain_trigger.should_retrain(retrain_trigger.read_alerts()) is False
