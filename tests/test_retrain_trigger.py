from src.monitoring.retrain_trigger import should_retrain


def test_no_alerts_no_retrain():
    assert should_retrain([]) is False


def test_small_psi_no_retrain():
    assert should_retrain([{"feature": "AGE", "psi": 0.05}]) is False


def test_large_psi_triggers_retrain():
    assert should_retrain([{"feature": "AGE", "psi": 0.4}]) is True


def test_threshold_is_inclusive():
    assert should_retrain([{"feature": "AGE", "psi": 0.25}], threshold=0.25) is True
