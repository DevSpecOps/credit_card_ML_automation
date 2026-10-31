import numpy as np
from src.monitoring.drift import psi


def test_psi_same_distribution():
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(0, 1, 1000)
    assert psi(expected, actual) < 0.1


def test_psi_high_drift():
    expected = np.random.normal(0, 1, 1000)
    actual = np.random.normal(5, 1, 1000)
    assert psi(expected, actual) > 0.25
