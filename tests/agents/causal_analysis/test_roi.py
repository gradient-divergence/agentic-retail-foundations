import pytest
import numpy as np
import math

# Functions to test
from agents.causal_analysis.roi import calculate_promotion_roi, interpret_causal_impact # Ensure both are imported

# --- Tests for calculate_promotion_roi ---

def test_calculate_roi_positive_scenario():
    """Test ROI calculation for a typical positive ATE scenario."""
    results = calculate_promotion_roi(
        estimated_ate=10.0,             # Sales uplift of 10 units per promotion
        average_baseline_sales=50.0,    # Average sales without promotion
        num_treated_units=100,          # 100 units received promotion
        promotion_cost_per_instance=2.0,# $2 cost per promotion
        margin_percent=0.25,            # 25% profit margin
        treatment_variable="promo_X"
    )
    assert results is not None
    assert results["estimated_ate"] == 10.0
    assert results["num_treated_units"] == 100
    assert results["average_baseline_sales"] == 50.0
    assert results["promotion_cost_per_instance"] == 2.0
    assert results["margin_percent"] == 0.25

    # total_sales_uplift = 10.0 * 100 = 1000.0
    assert results["total_sales_uplift"] == pytest.approx(1000.0)
    # profit_from_uplift = 1000.0 * 0.25 = 250.0
    assert results["profit_from_uplift"] == pytest.approx(250.0)
    # total_promotion_cost = 2.0 * 100 = 200.0
    assert results["total_promotion_cost"] == pytest.approx(200.0)
    # net_profit = 250.0 - 200.0 = 50.0
    assert results["net_profit"] == pytest.approx(50.0)
    # roi_percentage = (50.0 / 200.0) * 100 = 25.0
    assert results["roi_percentage"] == pytest.approx(25.0)

def test_calculate_roi_negative_scenario():
    """Test ROI calculation for a negative ROI scenario."""
    results = calculate_promotion_roi(
        estimated_ate=2.0,              # Low uplift
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=5.0, # High cost
        margin_percent=0.10,             # Low margin
        treatment_variable="promo_Y"
    )
    assert results is not None
    # total_sales_uplift = 2.0 * 100 = 200.0
    # profit_from_uplift = 200.0 * 0.10 = 20.0
    # total_promotion_cost = 5.0 * 100 = 500.0
    # net_profit = 20.0 - 500.0 = -480.0
    # roi_percentage = (-480.0 / 500.0) * 100 = -96.0
    assert results["net_profit"] == pytest.approx(-480.0)
    assert results["roi_percentage"] == pytest.approx(-96.0)

def test_calculate_roi_breakeven_scenario():
    """Test ROI calculation for a break-even scenario."""
    results = calculate_promotion_roi(
        estimated_ate=10.0,
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=2.5, # Cost adjusted for break-even
        margin_percent=0.25,
        treatment_variable="promo_Z"
    )
    assert results is not None
    # total_sales_uplift = 10.0 * 100 = 1000.0
    # profit_from_uplift = 1000.0 * 0.25 = 250.0
    # total_promotion_cost = 2.5 * 100 = 250.0
    # net_profit = 250.0 - 250.0 = 0.0
    # roi_percentage = (0.0 / 250.0) * 100 = 0.0
    assert results["net_profit"] == pytest.approx(0.0)
    assert results["roi_percentage"] == pytest.approx(0.0)

def test_calculate_roi_ate_is_none():
    """Test ROI calculation when estimated_ate is None."""
    results = calculate_promotion_roi(
        estimated_ate=None,
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=2.0,
        margin_percent=0.25,
        treatment_variable="promo_A"
    )
    assert results is None

def test_calculate_roi_zero_treated_units():
    """Test ROI calculation when num_treated_units is 0."""
    results = calculate_promotion_roi(
        estimated_ate=10.0,
        average_baseline_sales=50.0,
        num_treated_units=0,
        promotion_cost_per_instance=2.0,
        margin_percent=0.25,
        treatment_variable="promo_B"
    )
    assert results is not None
    assert results["error"] == "Number of treated units is non-positive."
    assert "estimated_ate" in results # Check other relevant info is passed back

def test_calculate_roi_negative_treated_units():
    """Test ROI calculation when num_treated_units is negative."""
    results = calculate_promotion_roi(
        estimated_ate=10.0,
        average_baseline_sales=50.0,
        num_treated_units=-5,
        promotion_cost_per_instance=2.0,
        margin_percent=0.25,
        treatment_variable="promo_C"
    )
    assert results is not None
    assert results["error"] == "Number of treated units is non-positive."

def test_calculate_roi_zero_cost_positive_profit():
    """Test ROI with zero promotion cost and positive net profit."""
    results = calculate_promotion_roi(
        estimated_ate=10.0,
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=0.0,
        margin_percent=0.25,
        treatment_variable="promo_D"
    )
    assert results is not None
    assert results["net_profit"] == pytest.approx(250.0) # 10*100*0.25
    assert results["roi_percentage"] == np.inf

def test_calculate_roi_zero_cost_zero_profit():
    """Test ROI with zero promotion cost and zero net profit."""
    results = calculate_promotion_roi(
        estimated_ate=0.0, # Zero uplift
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=0.0,
        margin_percent=0.25,
        treatment_variable="promo_E"
    )
    assert results is not None
    assert results["net_profit"] == pytest.approx(0.0)
    assert results["roi_percentage"] == 0.0

def test_calculate_roi_zero_cost_negative_profit():
    """Test ROI with zero promotion cost and negative net profit (e.g., negative ATE)."""
    results = calculate_promotion_roi(
        estimated_ate=-5.0, # Negative uplift
        average_baseline_sales=50.0,
        num_treated_units=100,
        promotion_cost_per_instance=0.0,
        margin_percent=0.25,
        treatment_variable="promo_F"
    )
    assert results is not None
    assert results["net_profit"] == pytest.approx(-125.0) # -5*100*0.25
    assert results["roi_percentage"] == 0.0

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