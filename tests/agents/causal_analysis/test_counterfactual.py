# tests/agents/causal_analysis/test_counterfactual.py

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Functions to test
from agents.causal_analysis.counterfactual import (
    fit_causal_forest_for_counterfactuals,
    perform_counterfactual_analysis,
    simulate_counterfactuals,
)  # Also test the wrapper

# --- Fixtures ---


@pytest.fixture
def cf_data() -> pd.DataFrame:
    """Sample DataFrame for counterfactual tests."""
    # Reuse estimator data structure
    return pd.DataFrame(
        {
            "sales": [100, 110, 105, 115, 200, 220, 210, 230],
            "promotion_applied": [0, 0, 0, 0, 1, 1, 1, 1],
            "price": [10, 10, 11, 11, 10, 10, 11, 11],
            "marketing": [50, 60, 50, 60, 55, 65, 55, 65],
            "non_numeric": ["A", "B", "A", "B", "A", "B", "A", "B"],
        }
    )


@pytest.fixture
def mock_causal_forest_model() -> MagicMock:
    """Fixture for a mocked fitted CausalForestDML model."""
    model = MagicMock(name="MockCausalForestDML")
    # Mock the methods used by simulate_counterfactuals
    # Use fixed random state for reproducible mock return values if needed
    rng = np.random.RandomState(0)
    model.effect.return_value = rng.rand(8) * 10  # Mock CATE predictions
    model.const_marginal_effect.return_value = rng.rand(8) * 10 + 100  # Mock constant effect
    return model


# --- Tests for fit_causal_forest_for_counterfactuals ---


@patch("agents.causal_analysis.counterfactual.GradientBoostingRegressor")
@patch("agents.causal_analysis.counterfactual.CausalForestDML")
def test_fit_causal_forest_success(mock_cf_dml, mock_gbr, cf_data):
    """Test successful fitting of Causal Forest."""
    mock_gbr_instance = MagicMock()
    mock_gbr.return_value = mock_gbr_instance
    mock_cf_dml_instance = MagicMock()
    mock_cf_dml.return_value = mock_cf_dml_instance

    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        n_estimators=50,  # Example kwarg
    )

    mock_gbr.assert_called()
    # Check specific kwargs passed to CausalForestDML constructor
    mock_cf_dml.assert_called_once_with(
        model_y=mock_gbr_instance,
        model_t=mock_gbr_instance,
        discrete_treatment=True,
        n_estimators=50,
        random_state=123,  # Default from function
    )
    mock_cf_dml_instance.fit.assert_called_once()
    # Check fit arguments if necessary (e.g., shapes of Y, T, X)
    # call_args, call_kwargs = mock_cf_dml_instance.fit.call_args
    # assert call_args[0].shape == (cf_data.shape[0],) # Y shape
    # assert call_args[1].shape == (cf_data.shape[0],) # T shape
    # assert call_kwargs['X'].shape == (cf_data.shape[0], 2) # X shape

    assert fitted_model is mock_cf_dml_instance


@patch("agents.causal_analysis.counterfactual.CausalForestDML", None)
def test_fit_causal_forest_not_installed(cf_data):
    """Test fitting returns None if EconML is not installed."""
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert fitted_model is None


def test_fit_causal_forest_missing_cols(cf_data):
    """Test fitting returns None when required columns are missing."""
    # The function catches the ValueError and should return None
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data.drop(columns=["sales"]),  # Drop outcome
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert fitted_model is None

    # Test missing treatment
    fitted_model_no_treat = fit_causal_forest_for_counterfactuals(
        data=cf_data.drop(columns=["promotion_applied"]),
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert fitted_model_no_treat is None

    # Test missing common cause
    fitted_model_no_cc = fit_causal_forest_for_counterfactuals(
        data=cf_data.drop(columns=["price"]),
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert fitted_model_no_cc is None


def test_fit_causal_forest_no_numeric_causes(cf_data):
    """Test fitting returns None if no numeric common causes are found."""
    data_no_numeric = cf_data[["sales", "promotion_applied", "non_numeric"]].copy()
    fitted_model = fit_causal_forest_for_counterfactuals(
        data=data_no_numeric,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["non_numeric"],  # Only non-numeric cause
    )
    assert fitted_model is None


@patch("agents.causal_analysis.counterfactual.CausalForestDML")
def test_fit_causal_forest_fit_exception(mock_cf_dml, cf_data):
    """Test fitting returns None if model.fit() raises exception."""
    mock_cf_dml_instance = MagicMock()
    mock_cf_dml_instance.fit.side_effect = RuntimeError("Fit failed")
    mock_cf_dml.return_value = mock_cf_dml_instance

    fitted_model = fit_causal_forest_for_counterfactuals(
        data=cf_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert fitted_model is None


# --- Tests for simulate_counterfactuals ---


def test_simulate_cf_set_treatment_1(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for setting treatment to 1."""
    scenario = {"set_treatment": 1}
    # Need to reset mocks if the fixture is reused across tests
    mock_causal_forest_model.reset_mock()
    # Redefine return values if needed for this specific test
    effect_val = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    const_effect_val = np.array([10, 11, 12, 13, 14, 15, 16, 17])
    mock_causal_forest_model.effect.return_value = effect_val
    mock_causal_forest_model.const_marginal_effect.return_value = const_effect_val

    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.effect.assert_called_once()
    mock_causal_forest_model.const_marginal_effect.assert_called_once()
    assert "factual_avg_effect" in results
    assert "counterfactual_avg_effect" in results
    assert "difference" in results
    # Check counterfactual avg effect matches the mocked const_marginal_effect mean
    assert results["factual_avg_effect"] == pytest.approx(np.mean(effect_val))
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(const_effect_val))
    assert results["difference"] == pytest.approx(np.mean(const_effect_val) - np.mean(effect_val))


def test_simulate_cf_set_treatment_0(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for setting treatment to 0."""
    scenario = {"set_treatment": 0}
    mock_causal_forest_model.reset_mock()
    effect_val = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    const_effect_val = np.array([20, 21, 22, 23, 24, 25, 26, 27])
    mock_causal_forest_model.effect.return_value = effect_val
    mock_causal_forest_model.const_marginal_effect.return_value = const_effect_val

    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.effect.assert_called_once()
    mock_causal_forest_model.const_marginal_effect.assert_called_once()
    # When T=0, counterfactual effect should still be based on const_marginal_effect
    assert results["factual_avg_effect"] == pytest.approx(np.mean(effect_val))
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(const_effect_val))
    assert results["difference"] == pytest.approx(np.mean(const_effect_val) - np.mean(effect_val))


def test_simulate_cf_adjust_feature(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation for adjusting a feature (partially implemented)."""
    scenario = {"adjust_feature": {"feature_name": "price", "value": 15}}
    mock_causal_forest_model.reset_mock()
    effect_val = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    mock_causal_forest_model.effect.return_value = effect_val

    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    assert results is not None
    assert results["scenario"] == scenario
    # Check that const_marginal_effect was NOT called
    mock_causal_forest_model.const_marginal_effect.assert_not_called()
    # Check difference is zero because counterfactual equals factual in placeholder logic
    assert results["difference"] == pytest.approx(0.0)
    assert results["factual_avg_effect"] == pytest.approx(np.mean(effect_val))
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(effect_val))


def test_simulate_cf_unsupported_scenario(mock_causal_forest_model, cf_data):
    """Test counterfactual simulation with an unsupported scenario."""
    scenario = {"unknown_action": "do_something"}
    mock_causal_forest_model.reset_mock()
    effect_val = np.array([2, 2, 2, 2, 2, 2, 2, 2])
    mock_causal_forest_model.effect.return_value = effect_val

    results = simulate_counterfactuals(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    assert results is not None
    assert results["scenario"] == scenario
    mock_causal_forest_model.const_marginal_effect.assert_not_called()
    # Expect factual = counterfactual
    assert results["difference"] == pytest.approx(0.0)
    assert results["factual_avg_effect"] == pytest.approx(np.mean(effect_val))
    assert results["counterfactual_avg_effect"] == pytest.approx(np.mean(effect_val))


def test_simulate_cf_no_model(cf_data):
    """Test simulation returns None if no model is provided."""
    scenario = {"set_treatment": 1}
    results = simulate_counterfactuals(
        model=None,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    assert results is None


def test_simulate_cf_model_method_missing(cf_data):
    """Test simulation handles AttributeError if model lacks required methods."""
    scenario = {"set_treatment": 1}
    # Create a mock that *doesn't* have the 'effect' method
    bad_model = MagicMock(spec=[])  # spec=[] ensures it has no default methods
    results = simulate_counterfactuals(
        model=bad_model,
        data=cf_data,
        treatment="promotion_applied",
        common_causes=["price", "marketing"],
        scenario=scenario,
    )
    # The function catches the AttributeError and returns None
    assert results is None


# --- Test for perform_counterfactual_analysis wrapper ---


@patch("agents.causal_analysis.counterfactual.simulate_counterfactuals")
def test_perform_counterfactual_analysis_wrapper(mock_simulate, mock_causal_forest_model, cf_data):
    """Test the wrapper function calls simulate_counterfactuals correctly."""
    scenario = {"set_treatment": 1}
    mock_simulate.return_value = {"test": "result"}  # Dummy return value

    result = perform_counterfactual_analysis(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promo",
        common_causes=["c1"],
        scenario=scenario,
    )

    mock_simulate.assert_called_once_with(
        model=mock_causal_forest_model,
        data=cf_data,
        treatment="promo",
        common_causes=["c1"],
        scenario=scenario,
    )
    assert result == {"test": "result"}
