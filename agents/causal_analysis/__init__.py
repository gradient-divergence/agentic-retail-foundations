"""Causal Analysis Tools for Retail Promotions"""

from .data_preparation import prepare_analysis_data
from .graph import define_causal_graph, visualize_causal_graph
from .estimators import (
    estimate_naive_ate,
    estimate_regression_ate,
    estimate_matching_ate,
    estimate_dowhy_ate,
    estimate_causalforest_ate,
    estimate_doubleml_irm_ate,
    run_dowhy_analysis, # Keep wrappers for now?
    run_double_ml_forest # Keep wrappers for now?
)
from .counterfactual import (
    fit_causal_forest_for_counterfactuals,
    simulate_counterfactuals,
    perform_counterfactual_analysis # Keep wrapper for now?
)
from .roi import calculate_promotion_roi, interpret_causal_impact


__all__ = [
    # Data Prep
    "prepare_analysis_data",
    # Graph
    "define_causal_graph",
    "visualize_causal_graph",
    # Estimators
    "estimate_naive_ate",
    "estimate_regression_ate",
    "estimate_matching_ate",
    "estimate_dowhy_ate",
    "estimate_causalforest_ate",
    "estimate_doubleml_irm_ate",
    # Counterfactuals
    "fit_causal_forest_for_counterfactuals",
    "simulate_counterfactuals",
    # ROI
    "calculate_promotion_roi",
    "interpret_causal_impact",
    # Wrappers (Consider removing later)
    "run_dowhy_analysis",
    "run_double_ml_forest",
    "perform_counterfactual_analysis",
] 