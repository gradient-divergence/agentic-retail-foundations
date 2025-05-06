# tests/agents/causal_analysis/test_estimators.py

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, ANY
import scipy.stats # Import scipy for mocking stats

# Functions to test
from agents.causal_analysis.estimators import (
    estimate_naive_ate,
    estimate_regression_ate,
    estimate_matching_ate, # Add matching estimator
    estimate_dowhy_ate, # Add dowhy estimator
    # estimate_causalforest_ate, # Add later
    # estimate_doubleml_irm_ate, # Add later
    _validate_input_data # Helper can be tested too
)

# --- Fixtures ---

@pytest.fixture
def estimator_data() -> pd.DataFrame:
    """Sample DataFrame for estimator tests."""
    # Simple data with clear expected outcomes
    return pd.DataFrame({
        'sales':           [100, 110, 105, 115, 200, 220, 210, 230], # Higher sales for treated
        'promotion_applied': [  0,   0,   0,   0,   1,   1,   1,   1], # Treatment indicator
        'price':           [ 10,  10,  11,  11,  10,  10,  11,  11], # Confounder 1
        'marketing':       [ 50,  60,  50,  60,  55,  65,  55,  65], # Confounder 2
        'non_numeric':     [ 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']  # To be ignored
    })

@pytest.fixture
def estimator_data_with_nans(estimator_data) -> pd.DataFrame:
    """Sample data with some NaNs."""
    data = estimator_data.copy()
    data.loc[1, 'sales'] = np.nan
    data.loc[5, 'price'] = np.nan
    return data

@pytest.fixture
def sample_graph_str() -> str:
    """A simple DOT graph string fixture."""
    return "digraph { price -> promotion_applied; marketing -> promotion_applied; price -> sales; marketing -> sales; promotion_applied -> sales; }"

# --- Tests for _validate_input_data helper ---

def test_validate_input_data_success(estimator_data):
    """Test helper validation passes with good data."""
    validated = _validate_input_data(
        estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )
    assert validated.shape[0] == estimator_data.shape[0]
    assert list(validated.columns) == ['sales', 'promotion_applied', 'price', 'marketing']

def test_validate_input_data_missing_col(estimator_data):
    """Test helper raises error if a required column is missing."""
    with pytest.raises(ValueError, match="missing required columns:.*'missing_cause'"):
        _validate_input_data(
            estimator_data,
            treatment='promotion_applied',
            outcome='sales',
            common_causes=['price', 'missing_cause']
        )

def test_validate_input_data_nan_drop(estimator_data_with_nans):
    """Test helper drops rows with NaNs in required columns."""
    validated = _validate_input_data(
        estimator_data_with_nans,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'] # price has NaN in row 5
    )
    # Original has 8 rows. Row 1 (NaN sales) and Row 5 (NaN price) should be dropped.
    assert validated.shape[0] == 6
    assert 1 not in validated.index
    assert 5 not in validated.index
    assert not validated.isnull().any().any()

# --- Tests for estimate_naive_ate ---

def test_estimate_naive_ate_success(estimator_data):
    """Test naive ATE calculation."""
    results = estimate_naive_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales'
    )
    control_mean = np.mean([100, 110, 105, 115]) # = 107.5
    treated_mean = np.mean([200, 220, 210, 230]) # = 215.0
    expected_ate = treated_mean - control_mean  # = 107.5

    assert results["control_mean"] == pytest.approx(control_mean)
    assert results["treated_mean"] == pytest.approx(treated_mean)
    assert results["naive_ate"] == pytest.approx(expected_ate)

def test_estimate_naive_ate_with_nans(estimator_data_with_nans):
    """Test naive ATE calculation handles NaNs by dropping rows."""
    # Row 1 (control, sales=NaN) is dropped.
    results = estimate_naive_ate(
        data=estimator_data_with_nans,
        treatment='promotion_applied',
        outcome='sales'
    )
    control_mean = np.mean([100, 105, 115]) # = 106.666...
    treated_mean = np.mean([200, 220, 210, 230]) # = 215.0 (no NaNs here)
    expected_ate = treated_mean - control_mean

    assert results["control_mean"] == pytest.approx(control_mean)
    assert results["treated_mean"] == pytest.approx(treated_mean)
    assert results["naive_ate"] == pytest.approx(expected_ate)

# --- Tests for estimate_regression_ate ---

# Mock statsmodels OLS and results
@patch('agents.causal_analysis.estimators.sm.OLS')
def test_estimate_regression_ate_success(mock_ols, estimator_data):
    """Test regression ATE estimation with mocks."""
    mock_results = MagicMock()
    mock_results.params = pd.Series({
        'const': 50.0,
        'promotion_applied': 105.0, # Expected ATE
        'price': -2.0,
        'marketing': 0.5
    })
    mock_results.pvalues = pd.Series({
        'const': 0.01,
        'promotion_applied': 0.001,
        'price': 0.1,
        'marketing': 0.04
    })
    # Create a dummy DataFrame for conf_int().loc[]
    conf_int_df = pd.DataFrame([[100.0, 110.0]], index=['promotion_applied'], columns=[0, 1])
    mock_results.conf_int.return_value = conf_int_df
    mock_results.summary.return_value.as_text.return_value = "Mock Summary"

    mock_ols_instance = MagicMock()
    mock_ols_instance.fit.return_value = mock_results
    mock_ols.return_value = mock_ols_instance

    results = estimate_regression_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )

    mock_ols.assert_called_once()
    mock_ols_instance.fit.assert_called_once()
    assert results["ate"] == pytest.approx(105.0)
    assert results["p_value"] == pytest.approx(0.001)
    assert results["conf_int"] == [100.0, 110.0]
    assert results["summary"] == "Mock Summary"

@patch('agents.causal_analysis.estimators.sm.OLS')
def test_estimate_regression_ate_ols_error(mock_ols, estimator_data):
    """Test regression ATE estimation handles OLS fit errors."""
    mock_ols_instance = MagicMock()
    mock_ols_instance.fit.side_effect = Exception("OLS failed")
    mock_ols.return_value = mock_ols_instance

    results = estimate_regression_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )

    assert "error" in results
    assert "OLS failed" in results["error"]

# --- Tests for estimate_matching_ate ---

@patch('scipy.stats.ttest_1samp') # Correct patch target
@patch('agents.causal_analysis.estimators.NearestNeighbors')
@patch('agents.causal_analysis.estimators.LogisticRegression')
@patch('agents.causal_analysis.estimators.StandardScaler')
def test_estimate_matching_ate_success(mock_scaler, mock_logit, mock_nn, mock_ttest, estimator_data):
    """Test successful PSM ATE estimation with mocks."""
    # --- Mock Setup ---
    # StandardScaler
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = estimator_data[['price', 'marketing']].values # Return dummy scaled data
    mock_scaler.return_value = mock_scaler_instance

    # LogisticRegression (Propensity Model)
    mock_logit_instance = MagicMock()
    # Return dummy probabilities (higher for treated group)
    mock_logit_instance.predict_proba.return_value = np.array([
        [0.8, 0.2], [0.7, 0.3], [0.8, 0.2], [0.7, 0.3], # Control (P(T=1))
        [0.3, 0.7], [0.2, 0.8], [0.3, 0.7], [0.2, 0.8]  # Treated (P(T=1))
    ])
    mock_logit.return_value = mock_logit_instance

    # NearestNeighbors
    mock_nn_instance = MagicMock()
    # Assume treated units are indices 4, 5, 6, 7
    # Assume control units are indices 0, 1, 2, 3
    # Mock kneighbors to return matches (e.g., treat 4 matches control 0, treat 5 matches control 1, etc.)
    # distances < caliper * std_dev (assume caliper allows these matches)
    mock_nn_instance.kneighbors.return_value = (
        np.array([[0.1], [0.1], [0.1], [0.1]]), # Distances
        np.array([[0], [1], [2], [3]])          # Indices of control units (relative to control_units df)
    )
    mock_nn.return_value = mock_nn_instance

    # Scipy stats ttest
    mock_ttest.return_value = (2.5, 0.04) # Mock t-statistic and p-value

    # --- Run Test ---
    results = estimate_matching_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'],
        caliper=0.25 # Example caliper
    )

    # --- Assertions ---
    # Check mocks were called
    mock_scaler_instance.fit_transform.assert_called_once()
    mock_logit_instance.fit.assert_called_once()
    mock_logit_instance.predict_proba.assert_called_once()
    mock_nn_instance.fit.assert_called_once()
    mock_nn_instance.kneighbors.assert_called_once()
    mock_ttest.assert_called_once() # Check ttest mock

    # Check results
    assert results["ate"] == pytest.approx(107.5) # (215 - 107.5) based on simple matching
    assert results["num_matched_treated"] == 4
    assert results["num_unmatched_treated"] == 0
    assert results["t_stat"] == pytest.approx(2.5)
    assert results["p_value"] == pytest.approx(0.04)

@patch('scipy.stats.ttest_1samp') # Correct patch target
@patch('agents.causal_analysis.estimators.NearestNeighbors')
@patch('agents.causal_analysis.estimators.LogisticRegression')
@patch('agents.causal_analysis.estimators.StandardScaler')
def test_estimate_matching_ate_no_matches(mock_scaler, mock_logit, mock_nn, mock_ttest, estimator_data):
    """Test PSM ATE estimation when no matches are found within caliper."""
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = estimator_data[['price', 'marketing']].values
    mock_scaler.return_value = mock_scaler_instance
    mock_logit_instance = MagicMock()
    mock_logit_instance.predict_proba.return_value = np.random.rand(8, 2) # Dummy probs
    mock_logit.return_value = mock_logit_instance
    mock_nn_instance = MagicMock()
    # Mock kneighbors to return distances larger than caliper * std_dev
    mock_nn_instance.kneighbors.return_value = (
        np.array([[0.5], [0.6], [0.7], [0.8]]), # Assume these distances are too large
        np.array([[0], [1], [2], [3]])
    )
    mock_nn.return_value = mock_nn_instance

    # Run test (caliper=0.1, assume std_dev makes 0.1 the threshold)
    # Need to mock np.std as well, or ensure caliper * std_dev < 0.5
    with patch('agents.causal_analysis.estimators.np.std', return_value=1.0):
         results = estimate_matching_ate(
            data=estimator_data,
            treatment='promotion_applied',
            outcome='sales',
            common_causes=['price', 'marketing'],
            caliper=0.1 # Set a small caliper
        )

    assert "error" in results
    assert "No matches found" in results["error"]
    mock_ttest.assert_not_called() # T-test shouldn't be called

def test_estimate_matching_ate_empty_groups(estimator_data):
    """Test PSM ATE estimation when treated or control group is empty."""
    # Test empty treated group (causes LogisticRegression error inside the function)
    data_only_control = estimator_data[estimator_data['promotion_applied'] == 0]
    results_no_treated = estimate_matching_ate(
        data=data_only_control,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )
    # Expect the function to catch the ValueError and return an error dict
    assert "error" in results_no_treated
    # Check if the error message from LogisticRegression is captured
    assert "solver needs samples of at least 2 classes" in results_no_treated["error"]

    # Test empty control group (also causes LogisticRegression error)
    data_only_treated = estimator_data[estimator_data['promotion_applied'] == 1]
    results_no_control = estimate_matching_ate(
        data=data_only_treated,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing']
    )
    # Expect the function to catch the ValueError and return an error dict
    assert "error" in results_no_control
    assert "solver needs samples of at least 2 classes" in results_no_control["error"]

# --- Tests for estimate_dowhy_ate ---

# Mock DoWhy's CausalModel
@patch('agents.causal_analysis.estimators.CausalModel')
def test_estimate_dowhy_ate_success(mock_causal_model, estimator_data, sample_graph_str):
    """Test successful DoWhy ATE estimation with mocks."""
    # --- Mock Setup ---
    mock_estimand = MagicMock()
    mock_estimate = MagicMock()
    mock_estimate.value = 106.5 # Expected ATE value

    mock_causal_model_instance = MagicMock()
    mock_causal_model_instance.identify_effect.return_value = mock_estimand
    mock_causal_model_instance.estimate_effect.return_value = mock_estimate
    mock_causal_model.return_value = mock_causal_model_instance

    # --- Run Test ---
    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'], # Provide numeric causes
        graph_str=sample_graph_str,
        method_name="backdoor.linear_regression"
    )

    # --- Assertions ---
    mock_causal_model.assert_called_once_with(
        data=ANY, # Check data argument passed (ignore specifics for now)
        treatment='promotion_applied',
        outcome='sales',
        graph=sample_graph_str,
        common_causes=['price', 'marketing']
    )
    # Check that the data passed to CausalModel has the expected shape/columns if needed
    call_args, call_kwargs = mock_causal_model.call_args
    passed_data = call_kwargs.get('data', call_args[0] if call_args else None)
    assert isinstance(passed_data, pd.DataFrame)
    assert list(passed_data.columns) == ['sales', 'promotion_applied', 'price', 'marketing'] # Validated data cols

    mock_causal_model_instance.identify_effect.assert_called_once_with(proceed_when_unidentified=True)
    mock_causal_model_instance.estimate_effect.assert_called_once_with(
        mock_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )
    assert ate == pytest.approx(106.5)


@patch('agents.causal_analysis.estimators.CausalModel', None) # Mock CausalModel as None
def test_estimate_dowhy_ate_not_installed(estimator_data, sample_graph_str):
    """Test DoWhy ATE estimation when library is not installed."""
    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'],
        graph_str=sample_graph_str
    )
    assert ate is None

@patch('agents.causal_analysis.estimators.CausalModel')
def test_estimate_dowhy_ate_estimation_error(mock_causal_model, estimator_data, sample_graph_str):
    """Test DoWhy ATE estimation handles errors during estimate_effect."""
    mock_causal_model_instance = MagicMock()
    mock_causal_model_instance.identify_effect.return_value = MagicMock()
    mock_causal_model_instance.estimate_effect.side_effect = Exception("DoWhy estimation failed")
    mock_causal_model.return_value = mock_causal_model_instance

    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing'],
        graph_str=sample_graph_str
    )
    assert ate is None # Should return None on error

@patch('agents.causal_analysis.estimators.CausalModel')
def test_estimate_dowhy_ate_non_numeric_causes(mock_causal_model, estimator_data, sample_graph_str):
    """Test DoWhy ATE estimation filters non-numeric common causes."""
    # --- Mock Setup (same as success case) ---
    mock_estimand = MagicMock()
    mock_estimate = MagicMock()
    mock_estimate.value = 106.5
    mock_causal_model_instance = MagicMock()
    mock_causal_model_instance.identify_effect.return_value = mock_estimand
    mock_causal_model_instance.estimate_effect.return_value = mock_estimate
    mock_causal_model.return_value = mock_causal_model_instance

    # --- Run Test --- with a non-numeric cause included
    ate = estimate_dowhy_ate(
        data=estimator_data, # Data includes 'non_numeric' column
        treatment='promotion_applied',
        outcome='sales',
        common_causes=['price', 'marketing', 'non_numeric'], # Include non-numeric cause
        graph_str=sample_graph_str,
    )

    # --- Assertions ---
    # Check that CausalModel was called with *only* the numeric common causes
    mock_causal_model.assert_called_once_with(
        data=ANY,
        treatment='promotion_applied',
        outcome='sales',
        graph=sample_graph_str,
        common_causes=['price', 'marketing'] # 'non_numeric' should be filtered out
    )
    assert ate == pytest.approx(106.5)

# --- Tests for estimate_causalforest_ate (To be added later) ---

# --- Tests for estimate_doubleml_irm_ate (To be added later) --- 