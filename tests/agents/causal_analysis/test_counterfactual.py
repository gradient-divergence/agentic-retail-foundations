# tests/agents/causal_analysis/test_counterfactual.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY

# Functions to test
from agents.causal_analysis.counterfactual import (
    fit_causal_forest_for_counterfactuals,
    simulate_counterfactuals,
    perform_counterfactual_analysis # Also test the wrapper
)

# --- Fixtures ---

@pytest.fixture
def cf_data() -> pd.DataFrame:
    """Sample DataFrame for counterfactual tests."""
    # Reuse estimator data structure
    return pd.DataFrame({
        'sales':           [100, 110, 105, 115, 200, 220, 210, 230],
        'promotion_applied': [  0,   0,   0,   0,   1,   1,   1,   1],
        'price':           [ 10,  10,  11,  11,  10,  10,  11,  11],
        'marketing':       [ 50,  60,  50,  60,  55,  65,  55,  65],
        'non_numeric':     [ 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
    })

@pytest.fixture
def mock_causal_forest_model() -> MagicMock:
    """Fixture for a mocked fitted CausalForestDML model."""
    model = MagicMock(name="MockCausalForestDML")
    # Mock the methods used by simulate_counterfactuals
    model.effect.return_value = np.random.rand(8) * 10 # Mock CATE predictions
    model.const_marginal_effect.return_value = np.random.rand(8) * 10 + 100 # Mock constant effect
    return model

# --- Tests for fit_causal_forest_for_counterfactuals ---

@patch('agents.causal_analysis.counterfactual.GradientBoostingRegressor')
@patch('agents.causal_analysis.counterfactual.CausalForestDML')
def test_fit_causal_forest_success(mock_cf_dml, mock_gbr, cf_data):
    """Test successful fitting of Causal Forest."""
    mock_gbr_instance = MagicMock()
    mock_gbr.return_value = mock_gbr_instance
    mock_cf_dml_instance = MagicMock()
    mock_cf_dml.return_value = mock_cf_dml_instance

    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'],
        n_estimators=50 # Example kwarg
    )

    mock_gbr.assert_called()
    mock_cf_dml.assert_called_once_with(
        model_y=mock_gbr_instance,
        model_t=mock_gbr_instance,
        discrete_treatment=True,
        n_estimators=50,
        random_state=123 # Default
    )
    mock_cf_dml_instance.fit.assert_called_once()
    assert fitted_model is mock_cf_dml_instance

@patch('agents.causal_analysis.counterfactual.CausalForestDML', None)
def test_fit_causal_forest_not_installed(cf_data):
    """Test fitting returns None if EconML is not installed."""
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )
    assert fitted_model is None

def test_fit_causal_forest_missing_cols(cf_data):
    """Test fitting returns None when required columns are missing."""
    # The function catches the ValueError and should return None
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data.drop(columns=['sales']),
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )
    assert fitted_model is None

def test_fit_causal_forest_no_numeric_causes(cf_data):
    """Test fitting returns None if no numeric common causes are found."""
    data_no_numeric = cf_data[['sales', 'promotion_applied', 'non_numeric']].copy()
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=data_no_numeric,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['non_numeric'] # Only non-numeric cause
    )
    assert fitted_model is None

# --- Tests for simulate_counterfactuals ---

def test_simulate_cf_set_treatment_1(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for setting treatment to 1."""
    scenario = {'set_treatment': 1}
    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.effect.assert_called_once()
    mock_causal_forest_model.const_marginal_effect.assert_called_once()
    assert "factual_avg_effect" in results
    assert "counterfactual_avg_effect" in results
    assert "difference" in results
    # Check counterfactual avg effect matches the mocked const_marginal_effect mean
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(mock_causal_forest_model.const_marginal_effect()))

def test_simulate_cf_set_treatment_0(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for setting treatment to 0."""
    scenario = {'set_treatment': 0}
    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.effect.assert_called_once()
    mock_causal_forest_model.const_marginal_effect.assert_called_once()
    # When T=0, counterfactual effect should still be based on const_marginal_effect
    # as per the simplified logic in the function.
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(mock_causal_forest_model.const_marginal_effect()))

def test_simulate_cf_adjust_feature(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for adjusting a feature (partially implemented)."""
    scenario = {'adjust_feature': {'feature_name': 'price', 'value': 15}}
    # Expecting a warning and default behavior due to partial implementation
    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    assert results is not None
    assert results["scenario"] == scenario
    # Check that const_marginal_effect was NOT called for this specific scenario type
    # based on current implementation (prints warning, uses factual as placeholder)
    mock_causal_forest_model.const_marginal_effect.assert_not_called()
    # Check difference is zero because counterfactual equals factual in placeholder
    assert results["difference"] == pytest.approx(0.0)

def test_simulate_cf_unsupported_scenario(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation with an unsupported scenario."""
    scenario = {'unknown_action': 'do_something'}
    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.const_marginal_effect.assert_not_called()
    # Expect factual = counterfactual
    assert results["difference"] == pytest.approx(0.0)

def test_simulate_cf_no_model(cf_data):
    """Test simulation returns None if no model is provided."""
    scenario = {'set_treatment': 1}
    results = simulate_counterfactuals(
        model=None,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    assert results is None

def test_simulate_cf_model_method_missing(cf_data):
    """Test simulation handles AttributeError if model lacks required methods."""
    scenario = {'set_treatment': 1}
    # Create a mock that *doesn't* have the 'effect' method
    bad_model = MagicMock(spec=[]) # spec=[] ensures it has no default methods
    results = simulate_counterfactuals(
        model=bad_model,
        data=cf_data,
        treatment='promotion_applied',
        common_causes=['price', 'marketing'],
        scenario=scenario
    )
    # The function catches the AttributeError and returns None
    assert results is None

# --- Test for perform_counterfactual_analysis wrapper ---

@patch('agents.causal_analysis.counterfactual.simulate_counterfactuals')
def test_perform_counterfactual_analysis_wrapper(mock_simulate, mock_causal_forest_model, cf_data):
    """Test the wrapper function calls simulate_counterfactuals correctly."""
    scenario = {'set_treatment': 1}
    mock_simulate.return_value = {"test": "result"} # Dummy return value

    result = perform_counterfactual_analysis(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promo",
        common_causes=["c1"],
        scenario=scenario
    )

    mock_simulate.assert_called_once_with(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promo",
        common_causes=["c1"],
        scenario=scenario
    )
    assert result == {"test": "result"} 