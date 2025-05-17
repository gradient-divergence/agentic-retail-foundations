# tests/agents/causal_analysis/test_estimators.py

from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Functions to test
from agents.causal_analysis.estimators import (
    _validate_input_data,
    estimate_causalforest_ate,
    estimate_doubleml_irm_ate,
    estimate_dowhy_ate,  # Add dowhy estimator
    estimate_matching_ate,
    estimate_naive_ate,
    estimate_regression_ate,
    run_double_ml_forest,
    run_dowhy_analysis,
)  # Helper can be tested too  # Add causal forest  # Add DoubleML estimator  # Add matching estimator  # Add run_double_ml_forest wrapper  # Add run_dowhy_analysis wrapper

# --- Fixtures ---


@pytest.fixture
def estimator_data() -> pd.DataFrame:
    """Sample DataFrame for estimator tests."""
    # Simple data with clear expected outcomes
    return pd.DataFrame(
        {
            "sales": [
                100,
                110,
                105,
                115,
                200,
                220,
                210,
                230,
            ],  # Higher sales for treated
            "promotion_applied": [0, 0, 0, 0, 1, 1, 1, 1],  # Treatment indicator
            "price": [10, 10, 11, 11, 10, 10, 11, 11],  # Confounder 1
            "marketing": [50, 60, 50, 60, 55, 65, 55, 65],  # Confounder 2
            "non_numeric": ["A", "B", "A", "B", "A", "B", "A", "B"],  # To be ignored
        }
    )


@pytest.fixture
def estimator_data_with_nans(estimator_data) -> pd.DataFrame:
    """Sample data with some NaNs."""
    data = estimator_data.copy()
    data.loc[1, "sales"] = np.nan
    data.loc[5, "price"] = np.nan
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
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert validated.shape[0] == estimator_data.shape[0]
    assert list(validated.columns) == [
        "sales",
        "promotion_applied",
        "price",
        "marketing",
    ]


def test_validate_input_data_missing_col(estimator_data):
    """Test helper raises error if a required column is missing."""
    with pytest.raises(ValueError, match="missing required columns:.*'missing_cause'"):
        _validate_input_data(
            estimator_data,
            treatment="promotion_applied",
            outcome="sales",
            common_causes=["price", "missing_cause"],
        )


def test_validate_input_data_nan_drop(estimator_data_with_nans):
    """Test helper drops rows with NaNs in required columns."""
    validated = _validate_input_data(
        estimator_data_with_nans,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],  # price has NaN in row 5
    )
    # Original has 8 rows. Row 1 (NaN sales) and Row 5 (NaN price) should be dropped.
    assert validated.shape[0] == 6
    assert 1 not in validated.index
    assert 5 not in validated.index
    assert not validated.isnull().any().any()


# --- Tests for estimate_naive_ate ---


def test_estimate_naive_ate_success(estimator_data):
    """Test naive ATE calculation."""
    results = estimate_naive_ate(data=estimator_data, treatment="promotion_applied", outcome="sales")
    control_mean = np.mean([100, 110, 105, 115])  # = 107.5
    treated_mean = np.mean([200, 220, 210, 230])  # = 215.0
    expected_ate = treated_mean - control_mean  # = 107.5

    assert results["control_mean"] == pytest.approx(control_mean)
    assert results["treated_mean"] == pytest.approx(treated_mean)
    assert results["naive_ate"] == pytest.approx(expected_ate)


def test_estimate_naive_ate_with_nans(estimator_data_with_nans):
    """Test naive ATE calculation handles NaNs by dropping rows."""
    # Row 1 (control, sales=NaN) is dropped.
    results = estimate_naive_ate(data=estimator_data_with_nans, treatment="promotion_applied", outcome="sales")
    control_mean = np.mean([100, 105, 115])  # = 106.666...
    treated_mean = np.mean([200, 220, 210, 230])  # = 215.0 (no NaNs here)
    expected_ate = treated_mean - control_mean

    assert results["control_mean"] == pytest.approx(control_mean)
    assert results["treated_mean"] == pytest.approx(treated_mean)
    assert results["naive_ate"] == pytest.approx(expected_ate)


# --- Tests for estimate_regression_ate ---


# Mock statsmodels OLS and results
@patch("agents.causal_analysis.estimators.sm.OLS")
def test_estimate_regression_ate_success(mock_ols, estimator_data):
    """Test regression ATE estimation with mocks."""
    mock_results = MagicMock()
    mock_results.params = pd.Series(
        {
            "const": 50.0,
            "promotion_applied": 105.0,  # Expected ATE
            "price": -2.0,
            "marketing": 0.5,
        }
    )
    mock_results.pvalues = pd.Series({"const": 0.01, "promotion_applied": 0.001, "price": 0.1, "marketing": 0.04})
    # Create a dummy DataFrame for conf_int().loc[]
    conf_int_df = pd.DataFrame([[100.0, 110.0]], index=["promotion_applied"], columns=[0, 1])
    mock_results.conf_int.return_value = conf_int_df
    mock_results.summary.return_value.as_text.return_value = "Mock Summary"

    mock_ols_instance = MagicMock()
    mock_ols_instance.fit.return_value = mock_results
    mock_ols.return_value = mock_ols_instance

    results = estimate_regression_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )

    mock_ols.assert_called_once()
    mock_ols_instance.fit.assert_called_once()
    assert results["ate"] == pytest.approx(105.0)
    assert results["p_value"] == pytest.approx(0.001)
    assert results["conf_int"] == [100.0, 110.0]
    assert results["summary"] == "Mock Summary"


@patch("agents.causal_analysis.estimators.sm.OLS")
def test_estimate_regression_ate_ols_error(mock_ols, estimator_data):
    """Test regression ATE estimation handles OLS fit errors."""
    mock_ols_instance = MagicMock()
    mock_ols_instance.fit.side_effect = Exception("OLS failed")
    mock_ols.return_value = mock_ols_instance

    results = estimate_regression_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )

    assert "error" in results
    assert "OLS failed" in results["error"]


# --- Tests for estimate_matching_ate ---


@patch("scipy.stats.ttest_1samp")  # Correct patch target
@patch("agents.causal_analysis.estimators.NearestNeighbors")
@patch("agents.causal_analysis.estimators.LogisticRegression")
@patch("agents.causal_analysis.estimators.StandardScaler")
def test_estimate_matching_ate_success(mock_scaler, mock_logit, mock_nn, mock_ttest, estimator_data):
    """Test successful PSM ATE estimation with mocks."""
    # --- Mock Setup ---
    # StandardScaler
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = estimator_data[["price", "marketing"]].values  # Return dummy scaled data
    mock_scaler.return_value = mock_scaler_instance

    # LogisticRegression (Propensity Model)
    mock_logit_instance = MagicMock()
    # Return dummy probabilities (higher for treated group)
    mock_logit_instance.predict_proba.return_value = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.8, 0.2],
            [0.7, 0.3],  # Control (P(T=1))
            [0.3, 0.7],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.2, 0.8],  # Treated (P(T=1))
        ]
    )
    mock_logit.return_value = mock_logit_instance

    # NearestNeighbors
    mock_nn_instance = MagicMock()
    # Assume treated units are indices 4, 5, 6, 7
    # Assume control units are indices 0, 1, 2, 3
    # Mock kneighbors to return matches (e.g., treat 4 matches control 0, treat 5 matches control 1, etc.)
    # distances < caliper * std_dev (assume caliper allows these matches)
    mock_nn_instance.kneighbors.return_value = (
        np.array([[0.1], [0.1], [0.1], [0.1]]),  # Distances
        np.array([[0], [1], [2], [3]]),  # Indices of control units (relative to control_units df)
    )
    mock_nn.return_value = mock_nn_instance

    # Scipy stats ttest
    mock_ttest.return_value = (2.5, 0.04)  # Mock t-statistic and p-value

    # --- Run Test ---
    results = estimate_matching_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        caliper=0.25,  # Example caliper
    )

    # --- Assertions ---
    # Check mocks were called
    mock_scaler_instance.fit_transform.assert_called_once()
    mock_logit_instance.fit.assert_called_once()
    mock_logit_instance.predict_proba.assert_called_once()
    mock_nn_instance.fit.assert_called_once()
    mock_nn_instance.kneighbors.assert_called_once()
    mock_ttest.assert_called_once()  # Check ttest mock

    # Check results
    assert results["ate"] == pytest.approx(107.5)  # (215 - 107.5) based on simple matching
    assert results["num_matched_treated"] == 4
    assert results["num_unmatched_treated"] == 0
    assert results["t_stat"] == pytest.approx(2.5)
    assert results["p_value"] == pytest.approx(0.04)


@patch("scipy.stats.ttest_1samp")  # Correct patch target
@patch("agents.causal_analysis.estimators.NearestNeighbors")
@patch("agents.causal_analysis.estimators.LogisticRegression")
@patch("agents.causal_analysis.estimators.StandardScaler")
def test_estimate_matching_ate_no_matches(mock_scaler, mock_logit, mock_nn, mock_ttest, estimator_data):
    """Test PSM ATE estimation when no matches are found within caliper."""
    mock_scaler_instance = MagicMock()
    mock_scaler_instance.fit_transform.return_value = estimator_data[["price", "marketing"]].values
    mock_scaler.return_value = mock_scaler_instance
    mock_logit_instance = MagicMock()
    mock_logit_instance.predict_proba.return_value = np.random.rand(8, 2)  # Dummy probs
    mock_logit.return_value = mock_logit_instance
    mock_nn_instance = MagicMock()
    # Mock kneighbors to return distances larger than caliper * std_dev
    mock_nn_instance.kneighbors.return_value = (
        np.array([[0.5], [0.6], [0.7], [0.8]]),  # Assume these distances are too large
        np.array([[0], [1], [2], [3]]),
    )
    mock_nn.return_value = mock_nn_instance

    # Run test (caliper=0.1, assume std_dev makes 0.1 the threshold)
    # Need to mock np.std as well, or ensure caliper * std_dev < 0.5
    with patch("agents.causal_analysis.estimators.np.std", return_value=1.0):
        results = estimate_matching_ate(
            data=estimator_data,
            treatment="promotion_applied",
            outcome="sales",
            common_causes=["price", "marketing"],
            caliper=0.1,  # Set a small caliper
        )

    assert "error" in results
    assert "No matches found" in results["error"]
    mock_ttest.assert_not_called()  # T-test shouldn't be called


def test_estimate_matching_ate_empty_groups(estimator_data):
    """Test PSM ATE estimation when treated or control group is empty."""
    # Test empty treated group (causes LogisticRegression error inside the function)
    data_only_control = estimator_data[estimator_data["promotion_applied"] == 0]
    results_no_treated = estimate_matching_ate(
        data=data_only_control,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    # Expect the function to catch the ValueError and return an error dict
    assert "error" in results_no_treated
    # Check if the error message from LogisticRegression is captured
    assert "solver needs samples of at least 2 classes" in results_no_treated["error"]

    # Test empty control group (also causes LogisticRegression error)
    data_only_treated = estimator_data[estimator_data["promotion_applied"] == 1]
    results_no_control = estimate_matching_ate(
        data=data_only_treated,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    # Expect the function to catch the ValueError and return an error dict
    assert "error" in results_no_control
    assert "solver needs samples of at least 2 classes" in results_no_control["error"]


# --- Tests for estimate_dowhy_ate ---


# Mock DoWhy's CausalModel
@patch("agents.causal_analysis.estimators.CausalModel")
def test_estimate_dowhy_ate_success(mock_causal_model, estimator_data, sample_graph_str):
    """Test successful DoWhy ATE estimation with mocks."""
    # --- Mock Setup ---
    mock_estimand = MagicMock()
    mock_estimate = MagicMock()
    mock_estimate.value = 106.5  # Expected ATE value

    mock_causal_model_instance = MagicMock()
    mock_causal_model_instance.identify_effect.return_value = mock_estimand
    mock_causal_model_instance.estimate_effect.return_value = mock_estimate
    mock_causal_model.return_value = mock_causal_model_instance

    # --- Run Test ---
    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],  # Provide numeric causes
        graph_str=sample_graph_str,
        method_name="backdoor.linear_regression",
    )

    # --- Assertions ---
    mock_causal_model.assert_called_once_with(
        data=ANY,  # Check data argument passed (ignore specifics for now)
        treatment="promotion_applied",
        outcome="sales",
        graph=sample_graph_str,
        common_causes=["price", "marketing"],
    )
    # Check that the data passed to CausalModel has the expected shape/columns if needed
    call_args, call_kwargs = mock_causal_model.call_args
    passed_data = call_kwargs.get("data", call_args[0] if call_args else None)
    assert isinstance(passed_data, pd.DataFrame)
    assert list(passed_data.columns) == [
        "sales",
        "promotion_applied",
        "price",
        "marketing",
    ]  # Validated data cols

    mock_causal_model_instance.identify_effect.assert_called_once_with(proceed_when_unidentified=True)
    mock_causal_model_instance.estimate_effect.assert_called_once_with(
        mock_estimand, method_name="backdoor.linear_regression", test_significance=True
    )
    assert ate == pytest.approx(106.5)


@patch("agents.causal_analysis.estimators.CausalModel", None)  # Mock CausalModel as None
def test_estimate_dowhy_ate_not_installed(estimator_data, sample_graph_str):
    """Test DoWhy ATE estimation when library is not installed."""
    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        graph_str=sample_graph_str,
    )
    assert ate is None


@patch("agents.causal_analysis.estimators.CausalModel")
def test_estimate_dowhy_ate_estimation_error(mock_causal_model, estimator_data, sample_graph_str):
    """Test DoWhy ATE estimation handles errors during estimate_effect."""
    mock_causal_model_instance = MagicMock()
    mock_causal_model_instance.identify_effect.return_value = MagicMock()
    mock_causal_model_instance.estimate_effect.side_effect = Exception("DoWhy estimation failed")
    mock_causal_model.return_value = mock_causal_model_instance

    ate = estimate_dowhy_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        graph_str=sample_graph_str,
    )
    assert ate is None  # Should return None on error


@patch("agents.causal_analysis.estimators.CausalModel")
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
        data=estimator_data,  # Data includes 'non_numeric' column
        treatment="promotion_applied",
        outcome="sales",
        common_causes=[
            "price",
            "marketing",
            "non_numeric",
        ],  # Include non-numeric cause
        graph_str=sample_graph_str,
    )

    # --- Assertions ---
    # Check that CausalModel was called with *only* the numeric common causes
    mock_causal_model.assert_called_once_with(
        data=ANY,
        treatment="promotion_applied",
        outcome="sales",
        graph=sample_graph_str,
        common_causes=["price", "marketing"],  # 'non_numeric' should be filtered out
    )
    assert ate == pytest.approx(106.5)


# --- Tests for estimate_causalforest_ate ---


# Mock EconML's CausalForestDML and its dependencies (like GradientBoostingRegressor)
@patch("agents.causal_analysis.estimators.GradientBoostingRegressor")
@patch("agents.causal_analysis.estimators.CausalForestDML")
def test_estimate_causalforest_ate_success(mock_cf_dml, mock_gbr, estimator_data):
    """Test successful CausalForest ATE estimation with mocks."""
    # --- Mock Setup ---
    mock_gbr_instance = MagicMock()
    mock_gbr.return_value = mock_gbr_instance

    mock_cf_dml_instance = MagicMock()
    mock_cf_dml_instance.ate.return_value = 104.2  # Expected ATE
    # Mock the interval method (return dummy interval)
    mock_cf_dml_instance.ate_interval.return_value = (100.0, 108.4)
    mock_cf_dml.return_value = mock_cf_dml_instance

    # --- Run Test ---
    ate = estimate_causalforest_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        n_estimators=50,  # Example param
        min_samples_leaf=5,  # Example param
    )

    # --- Assertions ---
    mock_gbr.assert_called()  # Check that nuisance models were initialized
    mock_cf_dml.assert_called_once_with(
        model_y=mock_gbr_instance,  # Check nuisance models passed
        model_t=mock_gbr_instance,
        discrete_treatment=True,
        n_estimators=50,
        min_samples_leaf=5,
        random_state=123,  # Default random state used in function
    )
    mock_cf_dml_instance.fit.assert_called_once()  # Check fit was called
    mock_cf_dml_instance.ate.assert_called_once()
    mock_cf_dml_instance.ate_interval.assert_called_once_with(X=ANY, alpha=0.05)

    assert ate == pytest.approx(104.2)


@patch("agents.causal_analysis.estimators.CausalForestDML", None)  # Mock CausalForestDML as None
def test_estimate_causalforest_ate_not_installed(estimator_data):
    """Test CausalForest ATE estimation when library is not installed."""
    ate = estimate_causalforest_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert ate is None


@patch("agents.causal_analysis.estimators.GradientBoostingRegressor")
@patch("agents.causal_analysis.estimators.CausalForestDML")
def test_estimate_causalforest_ate_fit_error(mock_cf_dml, mock_gbr, estimator_data):
    """Test CausalForest ATE estimation handles errors during fit."""
    mock_gbr.return_value = MagicMock()
    mock_cf_dml_instance = MagicMock()
    mock_cf_dml_instance.fit.side_effect = Exception("CFit failed")
    mock_cf_dml.return_value = mock_cf_dml_instance

    ate = estimate_causalforest_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert ate is None  # Should return None on error


def test_estimate_causalforest_ate_no_numeric_causes(estimator_data):
    """Test CausalForest ATE estimation handles case with no numeric causes."""
    # Create data with only non-numeric potential causes
    data_no_numeric = estimator_data[["sales", "promotion_applied", "non_numeric"]].copy()
    ate = estimate_causalforest_ate(
        data=data_no_numeric,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["non_numeric"],  # Pass only the non-numeric cause
    )
    # Expect None because the function filters out non-numeric causes and finds none left
    assert ate is None


# --- Tests for estimate_doubleml_irm_ate ---


@patch("agents.causal_analysis.estimators.RandomForestClassifier")
@patch("agents.causal_analysis.estimators.RandomForestRegressor")
@patch("agents.causal_analysis.estimators.DoubleMLIRM")
@patch("agents.causal_analysis.estimators.DoubleMLData")
def test_estimate_doubleml_irm_rf_success(mock_dml_data, mock_dml_irm, mock_rf_reg, mock_rf_clf, estimator_data):
    """Test successful DoubleML IRM (RandomForest) ATE estimation with mocks."""
    # --- Mock Setup ---
    mock_dml_data_instance = MagicMock()
    mock_dml_data.return_value = mock_dml_data_instance
    mock_rf_reg_instance = MagicMock()
    mock_rf_reg.return_value = mock_rf_reg_instance
    mock_rf_clf_instance = MagicMock()
    mock_rf_clf.return_value = mock_rf_clf_instance

    mock_dml_irm_instance = MagicMock()
    # Simplification: Mock coef_ directly as a list containing the float
    # The function accesses coef_[0], so this should work.
    mock_dml_irm_instance.coef_ = [102.8]  # Mock as a list directly
    mock_dml_irm_instance.summary = "Mock DoubleML Summary (RF)"
    mock_dml_irm.return_value = mock_dml_irm_instance

    # --- Run Test ---
    ate = estimate_doubleml_irm_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        ml_learner_name="RandomForest",
    )

    # --- Assertions ---
    mock_dml_data.assert_called_once_with(ANY, y_col="sales", d_cols="promotion_applied", x_cols=["price", "marketing"])
    mock_rf_reg.assert_called_once()
    mock_rf_clf.assert_called_once()
    mock_dml_irm.assert_called_once_with(mock_dml_data_instance, ml_g=mock_rf_reg_instance, ml_m=mock_rf_clf_instance)
    mock_dml_irm_instance.fit.assert_called_once()
    # No longer need to assert __getitem__ called
    assert ate == pytest.approx(102.8)


@patch("agents.causal_analysis.estimators.LogisticRegressionCV")
@patch("agents.causal_analysis.estimators.LassoCV")
@patch("agents.causal_analysis.estimators.DoubleMLIRM")
@patch("agents.causal_analysis.estimators.DoubleMLData")
def test_estimate_doubleml_irm_lasso_success(mock_dml_data, mock_dml_irm, mock_lasso, mock_logit_cv, estimator_data):
    """Test successful DoubleML IRM (Lasso) ATE estimation with mocks."""
    # --- Mock Setup ---
    mock_dml_data_instance = MagicMock()
    mock_dml_data.return_value = mock_dml_data_instance
    mock_lasso_instance = MagicMock()
    mock_lasso.return_value = mock_lasso_instance
    mock_logit_cv_instance = MagicMock()
    mock_logit_cv.return_value = mock_logit_cv_instance

    mock_dml_irm_instance = MagicMock()
    # Simplification: Mock coef_ directly as a list containing the float
    mock_dml_irm_instance.coef_ = [103.1]  # Mock as a list directly
    mock_dml_irm_instance.summary = "Mock DoubleML Summary (Lasso)"
    mock_dml_irm.return_value = mock_dml_irm_instance

    # --- Run Test ---
    ate = estimate_doubleml_irm_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        ml_learner_name="Lasso",
    )

    # --- Assertions ---
    mock_dml_data.assert_called_once()
    mock_lasso.assert_called_once()
    mock_logit_cv.assert_called_once()
    mock_dml_irm.assert_called_once_with(mock_dml_data_instance, ml_g=mock_lasso_instance, ml_m=mock_logit_cv_instance)
    mock_dml_irm_instance.fit.assert_called_once()
    # No longer need to assert __getitem__ called
    assert ate == pytest.approx(103.1)


@patch("agents.causal_analysis.estimators.DoubleMLData", None)  # Mock DoubleMLData as None
@patch("agents.causal_analysis.estimators.DoubleMLIRM", None)  # Mock DoubleMLIRM as None
def test_estimate_doubleml_irm_not_installed(estimator_data):
    """Test DoubleML IRM ATE estimation when library is not installed."""
    ate = estimate_doubleml_irm_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
    )
    assert ate is None


@patch("agents.causal_analysis.estimators.DoubleMLIRM")
@patch("agents.causal_analysis.estimators.DoubleMLData")
def test_estimate_doubleml_irm_fit_error(mock_dml_data, mock_dml_irm, estimator_data):
    """Test DoubleML IRM ATE estimation handles errors during fit."""
    mock_dml_data.return_value = MagicMock()
    mock_dml_irm_instance = MagicMock()
    mock_dml_irm_instance.fit.side_effect = Exception("DML fit failed")
    mock_dml_irm.return_value = mock_dml_irm_instance

    ate = estimate_doubleml_irm_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        ml_learner_name="RandomForest",  # Learner doesn't matter here
    )
    assert ate is None  # Should return None on error


def test_estimate_doubleml_irm_unsupported_learner(estimator_data):
    """Test DoubleML IRM ATE estimation with an unsupported learner name."""
    ate = estimate_doubleml_irm_ate(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        ml_learner_name="XGBoost",  # Unsupported
    )
    assert ate is None


def test_estimate_doubleml_irm_no_numeric_causes(estimator_data):
    """Test DoubleML IRM ATE estimation handles case with no numeric causes."""
    data_no_numeric = estimator_data[["sales", "promotion_applied", "non_numeric"]].copy()
    ate = estimate_doubleml_irm_ate(
        data=data_no_numeric,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["non_numeric"],
    )
    assert ate is None


# --- Tests for Wrapper Functions ---


@patch("agents.causal_analysis.estimators.estimate_dowhy_ate")
def test_run_dowhy_analysis_wrapper(mock_estimate_dowhy, estimator_data, sample_graph_str):
    """Test the run_dowhy_analysis wrapper function."""
    mock_estimate_dowhy.return_value = 0.75  # Mock the ATE value
    expected_method_name = "backdoor.propensity_score_matching"

    result = run_dowhy_analysis(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        graph_str=sample_graph_str,
        method_name=expected_method_name,
    )

    mock_estimate_dowhy.assert_called_once_with(
        estimator_data,
        "promotion_applied",
        "sales",
        ["price", "marketing"],
        sample_graph_str,
        expected_method_name,
    )
    assert result == {f"dowhy_{expected_method_name.split('.')[-1]}_ate": 0.75}


@patch("agents.causal_analysis.estimators.estimate_causalforest_ate")
def test_run_double_ml_forest_wrapper(mock_estimate_causalforest, estimator_data):
    """Test the run_double_ml_forest wrapper function."""
    # Note: The wrapper is named run_double_ml_forest but calls estimate_causalforest_ate
    mock_estimate_causalforest.return_value = 0.88
    extra_kwargs = {"n_estimators": 200}

    result = run_double_ml_forest(
        data=estimator_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing"],
        **extra_kwargs,
    )

    mock_estimate_causalforest.assert_called_once_with(
        estimator_data,
        "promotion_applied",
        "sales",
        ["price", "marketing"],
        **extra_kwargs,
    )
    assert result == {"causal_forest_ate": 0.88}
