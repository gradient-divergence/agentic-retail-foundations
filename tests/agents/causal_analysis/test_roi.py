import pytest
import numpy as np
import math

# Functions to test
from agents.causal_analysis.roi import calculate_promotion_roi, interpret_causal_impact # Ensure both are imported

# ... (calculate_promotion_roi tests remain the same) ...

# --- Tests for interpret_causal_impact ---

@pytest.fixture
def positive_roi_results() -> dict:
    return {
        "estimated_ate": 6.0,
        "net_profit": 200.0,
        "roi_percentage": 20.0,
    }

@pytest.fixture
def negative_roi_results() -> dict:
    return {
        "estimated_ate": 1.0,
        "net_profit": -800.0,
        "roi_percentage": -80.0,
    }

@pytest.fixture
def breakeven_roi_results() -> dict:
    return {
        "estimated_ate": 5.0,
        "net_profit": 0.0,
        "roi_percentage": 0.0,
    }

def test_interpret_positive_roi(positive_roi_results):
    """Test interpretation for positive ROI."""
    interpretation = interpret_causal_impact(positive_roi_results)
    assert interpretation.startswith("The analysis estimated an Average Treatment Effect (ATE) of 6.0000.")
    assert "positive financial impact" in interpretation
    assert "net profit of $200.00" in interpretation
    assert "Return on Investment (ROI) of 20.00%." in interpretation

def test_interpret_negative_roi(negative_roi_results):
    """Test interpretation for negative ROI."""
    interpretation = interpret_causal_impact(negative_roi_results)
    assert interpretation.startswith("The analysis estimated an Average Treatment Effect (ATE) of 1.0000.")
    assert "negative financial impact" in interpretation
    assert "net loss of $800.00" in interpretation
    assert "(ROI: -80.00%)." in interpretation

def test_interpret_breakeven_roi(breakeven_roi_results):
    """Test interpretation for break-even ROI."""
    interpretation = interpret_causal_impact(breakeven_roi_results)
    assert interpretation.startswith("The analysis estimated an Average Treatment Effect (ATE) of 5.0000.")
    assert "break-even scenario" in interpretation
    assert "net profit of $0.00" in interpretation
    assert "ROI of 0.00%." in interpretation

def test_interpret_none_results():
    """Test interpretation when input results are None."""
    interpretation = interpret_causal_impact(None)
    assert "could not be completed" in interpretation

def test_interpret_error_results():
    """Test interpretation when input results dictionary contains an error key."""
    error_results = {"error": "Something went wrong"}
    interpretation = interpret_causal_impact(error_results)
    assert "resulted in an error" in interpretation 