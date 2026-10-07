import numpy as np
import pytest

from demos.conformal_calibration_demo import conformal_interval, run_demo


def test_demo_has_finite_ninety_percent_bounds(capsys):
    run_demo()
    output = capsys.readouterr().out
    assert "inf" not in output
    assert "Lower bounds:" in output
    assert "Upper bounds:" in output


def test_finite_sample_interval_uses_the_correct_order_statistic():
    lower, upper = conformal_interval(np.arange(1, 10), np.zeros(9), np.array([20]), alpha=0.2)
    assert lower.tolist() == [12]
    assert upper.tolist() == [28]
    lower, upper = conformal_interval(np.arange(1, 9), np.zeros(8), np.array([20]), alpha=0.1)
    assert np.isneginf(lower).all()
    assert np.isposinf(upper).all()


@pytest.mark.parametrize("alpha", [0, 1, -0.1, float("nan")])
def test_invalid_coverage_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        conformal_interval(np.ones(3), np.zeros(3), np.zeros(1), alpha)


def test_empty_calibration_data_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        conformal_interval(np.array([]), np.array([]), np.zeros(1))
