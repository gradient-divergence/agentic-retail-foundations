import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
from typing import List, Dict, Any, Optional
import logging # Import logging for caplog usage

# Class to test
from agents.promotion_causal import PromotionCausalAnalyzer

# --- Fixtures ---

@pytest.fixture
def sample_sales_data() -> pd.DataFrame:
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02']),
        'store_id': [1, 2, 1],
        'product_id': [101, 101, 102],
        'sales': [10, 12, 5],
        'price': [1.0, 1.0, 2.0],
        'marketing_spend': [100, 150, 100]
    })

@pytest.fixture
def sample_promotion_data() -> pd.DataFrame:
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01']),
        'store_id': [1],
        'product_id': [101],
        'promotion_applied': [1]
    })

@pytest.fixture
def mock_prepared_data() -> pd.DataFrame:
    return pd.DataFrame({
        'sales': [10, 12, 5, 15],
        'promotion_applied': [1, 0, 0, 1],
        'price': [1.0, 1.0, 2.0, 1.5],
        'marketing_spend': [100, 150, 100, 120]
    })

@pytest.fixture
def mock_graph_definition() -> tuple:
    # graph_str, treatment, outcome, common_causes_found
    return (
        "digraph { promotion_applied -> sales; price -> sales; price -> promotion_applied; }",
        "promotion_applied",
        "sales",
        ["price", "marketing_spend"]
    )

# --- Tests for Initialization ---

@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.define_causal_graph')
def test_analyzer_initialization_success(
    mock_define_graph: MagicMock,
    mock_prepare_data: MagicMock,
    sample_sales_data: pd.DataFrame,
    sample_promotion_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test successful initialization of PromotionCausalAnalyzer."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition

    with caplog.at_level(logging.INFO):
        analyzer = PromotionCausalAnalyzer(
            sales_data=sample_sales_data,
            promotion_data=sample_promotion_data,
            treatment="promotion_applied",
            outcome="sales",
            default_common_causes=["price", "marketing_spend"]
        )

    mock_prepare_data.assert_called_once()
    call_kwargs = mock_prepare_data.call_args.kwargs
    pd.testing.assert_frame_equal(call_kwargs['sales_data'], sample_sales_data)
    assert call_kwargs['product_data'] is None
    assert call_kwargs['store_data'] is None
    pd.testing.assert_frame_equal(call_kwargs['promotion_data'], sample_promotion_data)

    pd.testing.assert_frame_equal(analyzer.analysis_data, mock_prepared_data)

    mock_define_graph.assert_called_once_with(
        analysis_data=mock_prepared_data,
        treatment="promotion_applied",
        outcome="sales",
        common_causes=["price", "marketing_spend"]
    )
    assert analyzer.causal_graph_str == mock_graph_definition[0]
    assert analyzer.common_causes == mock_graph_definition[3]
    assert analyzer.treatment == "promotion_applied"
    assert analyzer.outcome == "sales"
    assert analyzer.fitted_models == {}
    assert analyzer.last_run_results == {}
    assert "Initializing PromotionCausalAnalyzer..." in caplog.text
    assert "Data preparation successful." in caplog.text
    assert "Causal graph definition successful." in caplog.text
    assert "Initialization complete." in caplog.text

@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.define_causal_graph')
def test_analyzer_initialization_data_prep_fails(
    mock_define_graph: MagicMock,
    mock_prepare_data: MagicMock,
    sample_sales_data: pd.DataFrame,
    caplog
):
    """Test initialization when data preparation fails."""
    mock_prepare_data.side_effect = ValueError("Data prep error")

    with caplog.at_level(logging.ERROR):
        analyzer = PromotionCausalAnalyzer(
            sales_data=sample_sales_data
        )

    mock_prepare_data.assert_called_once()
    assert analyzer.analysis_data is None
    mock_define_graph.assert_not_called()
    assert analyzer.causal_graph_str is None
    assert analyzer.common_causes is None
    assert "ERROR during data preparation: Data prep error" in caplog.text
    assert "Initialization failed due to data preparation error." in caplog.text

@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.define_causal_graph')
def test_analyzer_initialization_graph_def_fails(
    mock_define_graph: MagicMock,
    mock_prepare_data: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    caplog
):
    """Test initialization when causal graph definition fails."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.side_effect = ValueError("Graph def error")

    with caplog.at_level(logging.ERROR):
        analyzer = PromotionCausalAnalyzer(
            sales_data=sample_sales_data
        )

    mock_prepare_data.assert_called_once()
    pd.testing.assert_frame_equal(analyzer.analysis_data, mock_prepared_data)
    mock_define_graph.assert_called_once()
    assert analyzer.causal_graph_str is None
    assert analyzer.common_causes is None
    assert "ERROR during causal graph definition: Graph def error" in caplog.text

# --- Tests for visualize_graph ---

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.visualize_causal_graph')
def test_visualize_graph_success(
    mock_visualize: MagicMock,
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame, 
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple
):
    """Test visualize_graph calls the underlying function when graph exists."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    
    assert analyzer.causal_graph_str == mock_graph_definition[0]
    analyzer.visualize_graph(save_path="test_graph.png")
    mock_visualize.assert_called_once_with(mock_graph_definition[0], "test_graph.png")

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.visualize_causal_graph')
def test_visualize_graph_no_graph_defined(
    mock_visualize: MagicMock,
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame, 
    mock_prepared_data: pd.DataFrame,
    caplog 
):
    """Test visualize_graph does nothing and logs if no graph is defined."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.side_effect = ValueError("Graph def error") 
    # Initialization logs are fine, we capture logs for the specific call below
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)

    assert analyzer.causal_graph_str is None
    caplog.clear() # Clear init logs
    with caplog.at_level(logging.ERROR):
        analyzer.visualize_graph()
    
    mock_visualize.assert_not_called()
    assert "Error: Causal graph is not available" in caplog.text

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
@patch('agents.promotion_causal.visualize_causal_graph')
def test_visualize_graph_visualization_error(
    mock_visualize: MagicMock,
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog 
):
    """Test visualize_graph handles errors from the visualization function via log."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    mock_visualize.side_effect = Exception("Plotting failed")
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)

    assert analyzer.causal_graph_str is not None
    caplog.clear() # Clear init logs
    with caplog.at_level(logging.ERROR):
        analyzer.visualize_graph()

    mock_visualize.assert_called_once()
    assert "Error during graph visualization call: Plotting failed" in caplog.text
    assert caplog.records[0].exc_info is not None

# --- Tests for run_all_analyses ---

@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_all_analyses_no_analysis_data(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    caplog 
):
    """Test run_all_analyses returns error if analysis_data is None."""
    mock_prepare_data.side_effect = ValueError("Initial data prep failed")
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data) 
    assert analyzer.analysis_data is None

    caplog.clear()
    results = analyzer.run_all_analyses()
    assert "error" in results
    assert "Analysis data not properly initialized" in results["error"]
    assert "Error: Analysis data not properly initialized" in caplog.text

ESTIMATOR_PATCHES = [
    patch('agents.promotion_causal.estimate_naive_ate', return_value={"naive_ate": 0.1}),
    patch('agents.promotion_causal.estimate_regression_ate', return_value={"ate": 0.2, "p_value": 0.01}),
    patch('agents.promotion_causal.estimate_matching_ate', return_value={"ate": 0.3, "p_value": 0.02}),
    patch('agents.promotion_causal.estimate_dowhy_ate', return_value=0.4), 
    patch('agents.promotion_causal.estimate_causalforest_ate', return_value=0.5),
    patch('agents.promotion_causal.estimate_doubleml_irm_ate', return_value=0.6)
]

def apply_patches(patches):
    def decorator(func):
        for p in reversed(patches):
            func = p(func)
        return func
    return decorator

@apply_patches(ESTIMATOR_PATCHES)
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_all_analyses_default_estimators_success(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_estimate_doubleml_irm_ate: MagicMock,
    mock_estimate_causalforest_ate: MagicMock,
    mock_estimate_dowhy_ate: MagicMock, 
    mock_estimate_matching_ate: MagicMock,
    mock_estimate_regression_ate: MagicMock,
    mock_estimate_naive_ate: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test run_all_analyses with default estimators successfully."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=['price'])

    assert analyzer.analysis_data is not None
    assert analyzer.common_causes == mock_graph_definition[3]
    assert analyzer.causal_graph_str == mock_graph_definition[0]

    caplog.clear()
    results = analyzer.run_all_analyses() 

    mock_estimate_naive_ate.assert_called_once_with(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome)
    mock_estimate_regression_ate.assert_called_once_with(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes)
    mock_estimate_matching_ate.assert_called_once_with(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes)
    mock_estimate_causalforest_ate.assert_called_once_with(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes)
    
    mock_estimate_dowhy_ate.assert_not_called()
    mock_estimate_doubleml_irm_ate.assert_not_called()

    assert "naive" in results
    assert results["naive"]["naive_ate"] == 0.1
    assert "regression" in results
    assert results["regression"]["ate"] == 0.2
    assert "matching" in results
    assert results["matching"]["ate"] == 0.3
    assert "causal_forest" in results
    assert results["causal_forest"]["ate"] == 0.5
    
    assert analyzer.last_run_results == results
    assert "Running Naive Estimator..." in caplog.text
    assert "Running Regression Adjustment..." in caplog.text
    assert "Running Propensity Score Matching..." in caplog.text
    assert "Running Causal Forest DML..." in caplog.text
    assert "--- Causal Analyses Complete ---" in caplog.text

@apply_patches(ESTIMATOR_PATCHES)
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_all_analyses_custom_estimators_success(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_estimate_doubleml_irm_ate: MagicMock,
    mock_estimate_causalforest_ate: MagicMock,
    mock_estimate_dowhy_ate: MagicMock,
    mock_estimate_matching_ate: MagicMock,
    mock_estimate_regression_ate: MagicMock,
    mock_estimate_naive_ate: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test run_all_analyses with a custom list of estimators, including DoWhy and DoubleML."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=['price', 'marketing'])

    custom_estimators_to_run = [
        'naive', 
        'dowhy_regression', 
        'dowhy_matching',   
        'doubleml_rf',      
        'doubleml_lasso'    
    ]
    caplog.clear()
    results = analyzer.run_all_analyses(run_estimators=custom_estimators_to_run)

    mock_estimate_naive_ate.assert_called_once()
    assert mock_estimate_dowhy_ate.call_count == 2
    mock_estimate_dowhy_ate.assert_any_call(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes, graph_str=analyzer.causal_graph_str, method_name="backdoor.linear_regression")
    mock_estimate_dowhy_ate.assert_any_call(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes, graph_str=analyzer.causal_graph_str, method_name="backdoor.propensity_score_matching")
    
    assert mock_estimate_doubleml_irm_ate.call_count == 2
    mock_estimate_doubleml_irm_ate.assert_any_call(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes, ml_learner_name="RandomForest")
    mock_estimate_doubleml_irm_ate.assert_any_call(data=analyzer.analysis_data, treatment=analyzer.treatment, outcome=analyzer.outcome, common_causes=analyzer.common_causes, ml_learner_name="Lasso")

    mock_estimate_regression_ate.assert_not_called()
    mock_estimate_matching_ate.assert_not_called()
    mock_estimate_causalforest_ate.assert_not_called()

    assert "naive" in results
    assert "dowhy_regression" in results
    assert results["dowhy_regression"]["ate"] == 0.4 
    assert "dowhy_matching" in results
    assert results["dowhy_matching"]["ate"] == 0.4   
    assert "doubleml_rf" in results
    assert results["doubleml_rf"]["ate"] == 0.6     
    assert "doubleml_lasso" in results
    assert results["doubleml_lasso"]["ate"] == 0.6  

    assert analyzer.last_run_results == results
    assert "Running DoWhy (Linear Regression)..." in caplog.text
    assert "Running DoWhy (Propensity Score Matching)..." in caplog.text
    assert "Running DoubleML IRM (Random Forest)..." in caplog.text
    assert "Running DoubleML IRM (Lasso)..." in caplog.text
    assert "--- Causal Analyses Complete ---" in caplog.text

@apply_patches(ESTIMATOR_PATCHES) 
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_all_analyses_missing_common_causes(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_estimate_doubleml_irm_ate: MagicMock, 
    mock_estimate_causalforest_ate: MagicMock,
    mock_estimate_dowhy_ate: MagicMock,
    mock_estimate_matching_ate: MagicMock,
    mock_estimate_regression_ate: MagicMock,
    mock_estimate_naive_ate: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple, 
    caplog
):
    """Test that estimators requiring common causes are skipped if none are available."""
    mock_prepare_data.return_value = mock_prepared_data
    graph_str_ok, treatment_ok, outcome_ok, _ = mock_graph_definition
    mock_define_graph.return_value = (graph_str_ok, treatment_ok, outcome_ok, []) 
    
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    assert analyzer.common_causes == [] 

    caplog.clear()
    results = analyzer.run_all_analyses()

    mock_estimate_naive_ate.assert_called_once() 
    mock_estimate_regression_ate.assert_not_called() 
    mock_estimate_matching_ate.assert_not_called()   
    mock_estimate_causalforest_ate.assert_not_called()
    mock_estimate_dowhy_ate.assert_not_called()      
    mock_estimate_doubleml_irm_ate.assert_not_called()

    assert "naive" in results
    assert "regression" in results and "error" in results["regression"] and "Common causes not available" in results["regression"]["error"]
    assert "matching" in results and "error" in results["matching"] and "Common causes not available" in results["matching"]["error"]
    assert "causal_forest" in results and "error" in results["causal_forest"] and "Common causes not available" in results["causal_forest"]["error"]
        
    assert "Skipping Regression Adjustment (no common causes)." in caplog.text
    assert "Skipping Propensity Score Matching (no common causes)." in caplog.text
    assert "Skipping Causal Forest DML (no common causes)." in caplog.text

@apply_patches(ESTIMATOR_PATCHES)
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_all_analyses_missing_graph(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_estimate_doubleml_irm_ate: MagicMock,
    mock_estimate_causalforest_ate: MagicMock,
    mock_estimate_dowhy_ate: MagicMock,
    mock_estimate_matching_ate: MagicMock,
    mock_estimate_regression_ate: MagicMock,
    mock_estimate_naive_ate: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple, 
    caplog
):
    """Test that DoWhy estimators are skipped if no graph is defined."""
    mock_prepare_data.return_value = mock_prepared_data
    _, treatment_ok, outcome_ok, common_causes_ok = mock_graph_definition
    mock_define_graph.return_value = (None, treatment_ok, outcome_ok, common_causes_ok) 
    
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=common_causes_ok)
    assert analyzer.causal_graph_str is None
    assert analyzer.common_causes == common_causes_ok 

    estimators_to_run = ['dowhy_regression', 'dowhy_matching', 'naive']
    caplog.clear()
    results = analyzer.run_all_analyses(run_estimators=estimators_to_run)

    mock_estimate_naive_ate.assert_called_once() 
    mock_estimate_dowhy_ate.assert_not_called() 

    assert "naive" in results
    assert "dowhy_regression" in results and "error" in results["dowhy_regression"] and "causal graph not available" in results["dowhy_regression"]["error"].lower()
    assert "dowhy_matching" in results and "error" in results["dowhy_matching"] and "causal graph not available" in results["dowhy_matching"]["error"].lower()
 
    assert "Skipping DoWhy (Linear Regression) (no causal graph)." in caplog.text
    assert "Skipping DoWhy (Propensity Score Matching) (no causal graph)." in caplog.text

@apply_patches(ESTIMATOR_PATCHES) 
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_all_analyses_estimator_exception(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_estimate_doubleml_irm_ate: MagicMock,
    mock_estimate_causalforest_ate: MagicMock,
    mock_estimate_dowhy_ate: MagicMock,
    mock_estimate_matching_ate: MagicMock,
    mock_estimate_regression_ate: MagicMock, 
    mock_estimate_naive_ate: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test run_all_analyses handles an exception from one estimator and continues."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    mock_estimate_regression_ate.side_effect = Exception("Regression failed spectacularly")

    caplog.clear()
    results = analyzer.run_all_analyses()

    mock_estimate_naive_ate.assert_called_once()      
    mock_estimate_regression_ate.assert_called_once() 
    mock_estimate_matching_ate.assert_called_once()   
    mock_estimate_causalforest_ate.assert_called_once()

    assert "naive" in results and "error" not in results["naive"]
    assert "regression" in results and "error" in results["regression"]
    assert "Regression failed spectacularly" in results["regression"]["error"]
    assert "matching" in results and "error" not in results["matching"] 
    assert "causal_forest" in results and "error" not in results["causal_forest"] 

    assert "ERROR running regression estimator: Regression failed spectacularly" in caplog.text
    assert "Running Naive Estimator..." in caplog.text
    assert "Running Propensity Score Matching..." in caplog.text
    assert "Running Causal Forest DML..." in caplog.text

# --- Tests for fit_model_for_counterfactuals ---

@patch('agents.promotion_causal.fit_causal_forest_for_counterfactuals')
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_fit_model_for_counterfactuals_success(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_fit_cf_model: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test successful fitting of a model for counterfactuals."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition 
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    mock_fitted_model_instance = MagicMock(name="FittedCausalForest")
    mock_fit_cf_model.return_value = mock_fitted_model_instance
    
    kwargs_for_fit = {"n_estimators": 150}
    caplog.clear()
    fitted_model = analyzer.fit_model_for_counterfactuals(model_key="causal_forest", **kwargs_for_fit)

    mock_fit_cf_model.assert_called_once_with(
        data=analyzer.analysis_data,
        treatment=analyzer.treatment,
        outcome=analyzer.outcome,
        common_causes=analyzer.common_causes,
        **kwargs_for_fit
    )
    assert fitted_model is mock_fitted_model_instance
    assert analyzer.fitted_models["causal_forest"] is mock_fitted_model_instance
    assert "Model 'causal_forest' fitted and stored." in caplog.text

@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_fit_model_for_counterfactuals_no_analysis_data(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    caplog
):
    """Test fit_model_for_counterfactuals when analysis_data is None."""
    mock_prepare_data.side_effect = ValueError("Initial data prep failed")
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    assert analyzer.analysis_data is None

    caplog.clear()
    fitted_model = analyzer.fit_model_for_counterfactuals()
    assert fitted_model is None
    assert "Error: Analysis data not initialized. Cannot fit model." in caplog.text

@patch('agents.promotion_causal.fit_causal_forest_for_counterfactuals') 
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data')
def test_fit_model_no_common_causes_for_causal_forest(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_fit_cf_model: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test fitting causal_forest fails if common_causes is None."""
    mock_prepare_data.return_value = mock_prepared_data
    graph_str_ok, treatment_ok, outcome_ok, _ = mock_graph_definition
    mock_define_graph.return_value = (graph_str_ok, treatment_ok, outcome_ok, None) 
    
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    assert analyzer.common_causes is None

    caplog.clear()
    fitted_model = analyzer.fit_model_for_counterfactuals(model_key="causal_forest")
    assert fitted_model is None
    mock_fit_cf_model.assert_not_called()
    assert "Error: Common causes not available. Cannot fit 'causal_forest' model." in caplog.text

@patch('agents.promotion_causal.fit_causal_forest_for_counterfactuals')
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_fit_model_fitter_exception(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_fit_cf_model: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test handling an exception from the model fitting function."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    mock_fit_cf_model.side_effect = Exception("CF fitting exploded")
    
    caplog.clear()
    fitted_model = analyzer.fit_model_for_counterfactuals(model_key="causal_forest")
    assert fitted_model is None
    mock_fit_cf_model.assert_called_once()
    assert "causal_forest" not in analyzer.fitted_models 
    assert "ERROR fitting causal_forest model: CF fitting exploded" in caplog.text
    assert caplog.records[0].exc_info is not None

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_fit_model_unsupported_key(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test attempting to fit an unsupported model_key."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    caplog.clear()
    fitted_model = analyzer.fit_model_for_counterfactuals(model_key="super_xgb_tree")
    assert fitted_model is None
    assert "super_xgb_tree" not in analyzer.fitted_models
    assert "Model type 'super_xgb_tree' not currently supported" in caplog.text

# --- Tests for run_counterfactual_analysis ---

@patch('agents.promotion_causal.simulate_counterfactuals')
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_counterfactual_analysis_success(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_simulate_counterfactuals: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test successful counterfactual analysis."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    mock_fitted_cf_model = MagicMock(name="PreviouslyFittedCFModel")
    analyzer.fitted_models["causal_forest"] = mock_fitted_cf_model
    
    scenario = {"set_treatment": 1}
    mock_simulation_results = {"ate_lift": 50.0, "original_outcome": 1000.0, "counterfactual_outcome": 1050.0}
    mock_simulate_counterfactuals.return_value = mock_simulation_results

    caplog.clear()
    results = analyzer.run_counterfactual_analysis(scenario=scenario, model_key="causal_forest")

    mock_simulate_counterfactuals.assert_called_once_with(
        model=mock_fitted_cf_model,
        data=analyzer.analysis_data,
        treatment=analyzer.treatment,
        common_causes=analyzer.common_causes,
        scenario=scenario
    )
    assert results == mock_simulation_results
    assert "Counterfactual Simulation Results:" in caplog.text
    assert "Ate Lift: 50.0000" in caplog.text 

@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_counterfactual_analysis_no_analysis_data(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    caplog
):
    """Test run_counterfactual_analysis when analysis_data is None."""
    mock_prepare_data.side_effect = ValueError("Initial data prep failed")
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    assert analyzer.analysis_data is None

    caplog.clear()
    results = analyzer.run_counterfactual_analysis(scenario={})
    assert results is None
    assert "Error: Analysis data not initialized. Cannot run counterfactuals." in caplog.text

@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_counterfactual_analysis_model_not_fitted(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test run_counterfactual_analysis when the specified model is not fitted."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    assert "my_missing_model" not in analyzer.fitted_models
    caplog.clear()
    results = analyzer.run_counterfactual_analysis(scenario={}, model_key="my_missing_model")
    assert results is None
    assert "Error: Model 'my_missing_model' not found or not fitted." in caplog.text

@patch('agents.promotion_causal.simulate_counterfactuals')
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_counterfactual_analysis_simulation_error(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_simulate_counterfactuals: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test run_counterfactual_analysis when simulate_counterfactuals raises an error."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])

    mock_fitted_model = MagicMock(name="SomeModel")
    analyzer.fitted_models["causal_forest"] = mock_fitted_model
    mock_simulate_counterfactuals.side_effect = Exception("Simulation exploded")

    caplog.clear()
    results = analyzer.run_counterfactual_analysis(scenario={}, model_key="causal_forest")
    assert results is None
    mock_simulate_counterfactuals.assert_called_once()
    assert "ERROR during counterfactual simulation call: Simulation exploded" in caplog.text
    assert caplog.records[0].exc_info is not None

@patch('agents.promotion_causal.simulate_counterfactuals')
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_counterfactual_analysis_no_common_causes(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_simulate_counterfactuals: MagicMock, 
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test counterfactual analysis prints error if common_causes is None for relevant models."""
    mock_prepare_data.return_value = mock_prepared_data
    graph_str_ok, treatment_ok, outcome_ok, _ = mock_graph_definition
    mock_define_graph.return_value = (graph_str_ok, treatment_ok, outcome_ok, None) 
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    
    analyzer.fitted_models["causal_forest"] = MagicMock()
    assert analyzer.common_causes is None

    caplog.clear()
    results = analyzer.run_counterfactual_analysis(scenario={}, model_key="causal_forest")
    assert results is None 
    mock_simulate_counterfactuals.assert_not_called() 
    assert "Error: Common causes not available, cannot run counterfactuals for 'causal_forest'" in caplog.text

# --- Tests for run_roi_analysis ---

@patch('agents.promotion_causal.interpret_causal_impact')
@patch('agents.promotion_causal.calculate_promotion_roi')
@patch('agents.promotion_causal.define_causal_graph') 
@patch('agents.promotion_causal.prepare_analysis_data') 
def test_run_roi_analysis_success_from_ate_source(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_calculate_roi: MagicMock,
    mock_interpret_impact: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test successful ROI analysis using ATE from last_run_results."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    assert analyzer.analysis_data is not None 

    analyzer.last_run_results = {
        "causal_forest": {"ate": 0.5, "p_value": 0.01},
        "regression": {"ate": 0.45}
    }
    mock_roi_calculation_dict = {"roi_percentage": 25.0, "net_profit": 500.0, "estimated_ate": 0.5}
    mock_calculate_roi.return_value = mock_roi_calculation_dict
    mock_interpret_impact.return_value = "Promotion was very profitable."

    analyzer.analysis_data[analyzer.treatment] = [0,0,1,1] 
    analyzer.analysis_data[analyzer.outcome] = [10,10,15,15]    

    caplog.clear()
    results = analyzer.run_roi_analysis(
        ate_source="causal_forest", 
        promotion_cost_per_instance=1.0, 
        margin_percent=0.2
    )

    expected_num_treated = analyzer.analysis_data[analyzer.treatment].sum()
    control_mask = (analyzer.analysis_data[analyzer.treatment] == 0)
    expected_avg_baseline = analyzer.analysis_data.loc[control_mask, analyzer.outcome].mean()

    mock_calculate_roi.assert_called_once_with(
        estimated_ate=0.5,
        average_baseline_sales=expected_avg_baseline,
        num_treated_units=expected_num_treated,
        promotion_cost_per_instance=1.0,
        margin_percent=0.2,
        treatment_variable=analyzer.treatment
    )
    mock_interpret_impact.assert_called_once_with(mock_roi_calculation_dict)
    assert results is not None
    assert results["roi_calculation"] == mock_roi_calculation_dict
    assert results["interpretation"] == "Promotion was very profitable."
    assert "Using ATE=0.5000 from 'causal_forest' for ROI calculation." in caplog.text
    assert "ROI Interpretation: Promotion was very profitable." in caplog.text

@patch('agents.promotion_causal.interpret_causal_impact')
@patch('agents.promotion_causal.calculate_promotion_roi')
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_success_direct_ate(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_calculate_roi: MagicMock,
    mock_interpret_impact: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test successful ROI analysis using directly provided ATE."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    assert analyzer.analysis_data is not None

    direct_ate = 0.75
    mock_roi_calculation_dict = {"roi_percentage": 30.0, "net_profit": 600.0, "estimated_ate": direct_ate}
    mock_calculate_roi.return_value = mock_roi_calculation_dict
    mock_interpret_impact.return_value = "Direct ATE showed good profit."
    
    analyzer.analysis_data[analyzer.treatment] = [0,1,0,1] 
    analyzer.analysis_data[analyzer.outcome] = [10,20,12,22]

    caplog.clear()
    results = analyzer.run_roi_analysis(
        estimated_ate=direct_ate, 
        promotion_cost_per_instance=0.5, 
        margin_percent=0.3
    )
    
    expected_num_treated = analyzer.analysis_data[analyzer.treatment].sum()
    control_mask = (analyzer.analysis_data[analyzer.treatment] == 0)
    expected_avg_baseline = analyzer.analysis_data.loc[control_mask, analyzer.outcome].mean()

    mock_calculate_roi.assert_called_once_with(
        estimated_ate=direct_ate,
        average_baseline_sales=expected_avg_baseline,
        num_treated_units=expected_num_treated,
        promotion_cost_per_instance=0.5,
        margin_percent=0.3,
        treatment_variable=analyzer.treatment
    )
    mock_interpret_impact.assert_called_once_with(mock_roi_calculation_dict)
    assert results is not None
    assert results["roi_calculation"] == mock_roi_calculation_dict
    assert results["interpretation"] == "Direct ATE showed good profit."
    assert "ROI Interpretation: Direct ATE showed good profit." in caplog.text

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_no_analysis_data(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    caplog
):
    """Test run_roi_analysis when analysis_data is None."""
    mock_prepare_data.side_effect = ValueError("Data prep error")
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    assert analyzer.analysis_data is None

    caplog.clear()
    results = analyzer.run_roi_analysis(estimated_ate=0.5) 
    assert results is None
    assert "Error: Analysis data not available for ROI calculation." in caplog.text

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_ate_source_not_found(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test ROI analysis when ate_source is not in last_run_results."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    analyzer.last_run_results = {"some_other_method": {"ate": 0.1}} 
    analyzer.analysis_data[analyzer.treatment] = [0,1,0,1]
    analyzer.analysis_data[analyzer.outcome] = [10,20,12,22]

    caplog.clear()
    results = analyzer.run_roi_analysis(ate_source="non_existent_method")
    assert results is None
    assert "Error: Result for ATE source 'non_existent_method' not found" in caplog.text
    assert "Available results: ['some_other_method']" in caplog.text

@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_ate_extraction_error(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test ROI analysis when ATE cannot be extracted from last_run_results."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    analyzer.analysis_data[analyzer.treatment] = [0,1,0,1]
    analyzer.analysis_data[analyzer.outcome] = [10,20,12,22]

    caplog.clear()
    analyzer.last_run_results = {"bad_method": {"wrong_key": 0.1}} 
    results_wrong_key = analyzer.run_roi_analysis(ate_source="bad_method")
    assert results_wrong_key is None
    assert "Error: Could not extract a valid numeric ATE value" in caplog.text

    caplog.clear()
    analyzer.last_run_results = {"error_method": {"error": "previous estimation failure"}}
    results_prev_error = analyzer.run_roi_analysis(ate_source="error_method")
    assert results_prev_error is None
    assert "Error: Cannot use ATE from 'error_method' due to previous error: previous estimation failure" in caplog.text

@patch('agents.promotion_causal.interpret_causal_impact') 
@patch('agents.promotion_causal.calculate_promotion_roi') 
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_stat_calc_key_error(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_calculate_roi: MagicMock, 
    mock_interpret_impact: MagicMock, 
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame, 
    mock_graph_definition: tuple,
    caplog
):
    """Test ROI analysis handles KeyError if treatment/outcome cols are missing for stat calculation."""
    valid_prepared_data = mock_prepared_data.copy()
    valid_prepared_data["promotion_applied"] = [0, 1, 0, 1] 
    valid_prepared_data["sales"] = [10,15,12,18]      
    
    mock_prepare_data.return_value = valid_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, treatment="promotion_applied", outcome="sales", default_common_causes=mock_graph_definition[3])
        
    analysis_data_no_treatment = analyzer.analysis_data.copy().drop(columns=[analyzer.treatment])
    analyzer.analysis_data = analysis_data_no_treatment 
    
    caplog.clear()
    results_no_treat = analyzer.run_roi_analysis(estimated_ate=0.5)
    assert results_no_treat is None
    assert f"Error calculating num_treated_units: Treatment column '{analyzer.treatment}' not found" in caplog.text
    mock_calculate_roi.assert_not_called()

    analyzer.analysis_data = valid_prepared_data.copy() 
    analyzer.analysis_data[analyzer.treatment] = [0,0,1,1] 

    analysis_data_no_outcome = analyzer.analysis_data.copy().drop(columns=[analyzer.outcome])
    analyzer.analysis_data = analysis_data_no_outcome 

    caplog.clear()
    results_no_outcome = analyzer.run_roi_analysis(estimated_ate=0.5)
    assert results_no_outcome is None
    assert f"Error calculating average_baseline_sales: Outcome column '{analyzer.outcome}' not found" in caplog.text
    mock_calculate_roi.reset_mock() # Reset as it wasn't called in previous sub-test
    mock_calculate_roi.assert_not_called() # Still should not be called


@patch('agents.promotion_causal.interpret_causal_impact', side_effect=Exception("Interpret error"))
@patch('agents.promotion_causal.calculate_promotion_roi')
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_interpretation_exception(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_calculate_roi: MagicMock,
    mock_interpret_impact_exception: MagicMock, 
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test ROI analysis handles exception from interpret_causal_impact."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    analyzer.analysis_data[analyzer.treatment] = [0,1,0,1] 
    analyzer.analysis_data[analyzer.outcome] = [10,20,10,15]

    mock_roi_dict = {"roi_percentage": 10.0, "estimated_ate": 0.1, "net_profit": 100}
    mock_calculate_roi.return_value = mock_roi_dict
    
    caplog.clear()
    results = analyzer.run_roi_analysis(estimated_ate=0.1)
    assert results is not None
    assert "error" in results
    assert "ROI calculation failed: Interpret error" in results["error"]
    assert results["roi_calculation"] is None 
    assert results["interpretation"] is None

    mock_calculate_roi.assert_called_once()
    mock_interpret_impact_exception.assert_called_once_with(mock_roi_dict)
    assert "ERROR during ROI calculation/interpretation call: Interpret error" in caplog.text
    assert caplog.records[0].exc_info is not None

@patch('agents.promotion_causal.interpret_causal_impact')
@patch('agents.promotion_causal.calculate_promotion_roi', side_effect=Exception("Calc ROI error"))
@patch('agents.promotion_causal.define_causal_graph')
@patch('agents.promotion_causal.prepare_analysis_data')
def test_run_roi_analysis_calculation_exception(
    mock_prepare_data: MagicMock,
    mock_define_graph: MagicMock,
    mock_calculate_roi_exception: MagicMock,
    mock_interpret_impact: MagicMock,
    sample_sales_data: pd.DataFrame,
    mock_prepared_data: pd.DataFrame,
    mock_graph_definition: tuple,
    caplog
):
    """Test ROI analysis handles exception from calculate_promotion_roi."""
    mock_prepare_data.return_value = mock_prepared_data
    mock_define_graph.return_value = mock_graph_definition
    analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data, default_common_causes=mock_graph_definition[3])
    analyzer.analysis_data[analyzer.treatment] = [0,1,0,1]
    analyzer.analysis_data[analyzer.outcome] = [10,20,10,15]
    
    caplog.clear()
    results = analyzer.run_roi_analysis(estimated_ate=0.1)
    assert results is not None
    assert "error" in results
    assert "ROI calculation failed: Calc ROI error" in results["error"]
    assert results["roi_calculation"] is None
    assert results["interpretation"] is None 

    mock_calculate_roi_exception.assert_called_once()
    mock_interpret_impact.assert_not_called() 
    assert "ERROR during ROI calculation/interpretation call: Calc ROI error" in caplog.text
    assert caplog.records[0].exc_info is not None

# --- Tests for _is_holiday (example helper) ---

def test_is_holiday_helper(sample_sales_data, caplog): 
    # Minimal init just to get an analyzer instance
    with patch('agents.promotion_causal.prepare_analysis_data', return_value=pd.DataFrame()), \
         patch('agents.promotion_causal.define_causal_graph', return_value=("g", "t", "o", [])):
        analyzer = PromotionCausalAnalyzer(sales_data=sample_sales_data)
    
    caplog.clear()
    # Test with dates that should match the placeholder holidays
    dates_with_holidays = pd.Series(pd.to_datetime([
        '2023-01-01', '2023-01-02', '2023-07-04', '2023-12-20', '2023-12-25', '2024-01-01'
    ]))
    expected_holidays = pd.Series([True, False, True, False, True, False], index=dates_with_holidays.index)
    result_holidays = analyzer._is_holiday(dates_with_holidays)
    pd.testing.assert_series_equal(result_holidays, expected_holidays, check_dtype=False)
    assert not caplog.text # No warnings expected for valid dates

    caplog.clear() 
    dates_invalid = pd.Series(["not-a-date", "2023-01-01T00:00:00"])
    expected_invalid = pd.Series([False, False], index=dates_invalid.index)
    result_invalid = analyzer._is_holiday(dates_invalid)
    pd.testing.assert_series_equal(result_invalid, expected_invalid, check_dtype=False)
    assert "Warning: Failed to check holidays" in caplog.text

    caplog.clear()
    empty_dates = pd.Series([], dtype='datetime64[ns]')
    expected_empty = pd.Series([], dtype='bool')
    result_empty = analyzer._is_holiday(empty_dates)
    pd.testing.assert_series_equal(result_empty, expected_empty, check_dtype=False)
    assert not caplog.text # No warnings for empty series