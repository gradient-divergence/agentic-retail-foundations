# agents/promotion_causal.py

"""
PromotionCausalAnalyzer: Orchestrates causal inference and counterfactual analysis for retail promotions.

This module provides the PromotionCausalAnalyzer class, which uses specialized modules
for data preparation, graph definition, estimation, counterfactuals, and ROI calculation.
"""

import logging  # Import logging
from typing import Any

import pandas as pd

# Import from the new causal_analysis package
# Ensure this relative import works based on how the project is structured/run.
# If running scripts from the root, might need 'from agents.causal_analysis import ...'
try:
    from .causal_analysis import (
        calculate_promotion_roi,
        define_causal_graph,
        estimate_causalforest_ate,
        estimate_doubleml_irm_ate,
        estimate_dowhy_ate,
        estimate_matching_ate,
        estimate_naive_ate,
        estimate_regression_ate,
        fit_causal_forest_for_counterfactuals,
        interpret_causal_impact,
        prepare_analysis_data,
        simulate_counterfactuals,
        visualize_causal_graph,
    )
except ImportError:
    # Fallback for running script directly or different project structure
    print("Attempting absolute import for causal_analysis package...")
    from agents.causal_analysis import (
        calculate_promotion_roi,
        define_causal_graph,
        estimate_causalforest_ate,
        estimate_doubleml_irm_ate,
        estimate_dowhy_ate,
        estimate_matching_ate,
        estimate_naive_ate,
        estimate_regression_ate,
        fit_causal_forest_for_counterfactuals,
        interpret_causal_impact,
        prepare_analysis_data,
        simulate_counterfactuals,
        visualize_causal_graph,
    )


# Optional dependencies needed by the orchestrator itself (e.g., storing fitted models)
# CausalForestDML is used for type hinting if available
try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None  # Allow storing CausalForest if available, but don't require


class PromotionCausalAnalyzer:
    """
    Orchestrates the causal analysis of promotions on sales performance.

    Uses helper modules for specific tasks like data prep, estimation, and ROI.

    Attributes:
        sales_data_raw: Raw input sales data.
        product_data_raw: Raw input product data.
        store_data_raw: Raw input store data.
        promotion_data_raw: Raw input promotion data.
        treatment: Name of the treatment variable.
        outcome: Name of the outcome variable.
        analysis_data: Prepared DataFrame for analysis (result of prepare_analysis_data).
        causal_graph_str: The causal graph definition in DOT format string.
        common_causes: List of common cause variable names used in the analysis.
        fitted_models: Dictionary storing fitted models (e.g., for counterfactuals).
                       Keys are model identifiers (e.g., 'causal_forest').
        last_run_results: Dictionary storing the results from the last call to run_all_analyses.
    """

    def __init__(
        self,
        sales_data: pd.DataFrame,
        product_data: pd.DataFrame | None = None,
        store_data: pd.DataFrame | None = None,
        promotion_data: pd.DataFrame | None = None,
        treatment: str = "promotion_applied",
        outcome: str = "sales",
        default_common_causes: list[str] | None = None,
    ):
        """
        Initialize with retail datasets and configuration.

        Args:
            sales_data: DataFrame with sales transactions.
            product_data: Optional DataFrame with product features.
            store_data: Optional DataFrame with store features.
            promotion_data: Optional DataFrame with promotion details.
            treatment: Name of the treatment variable column.
            outcome: Name of the outcome variable column.
            default_common_causes: Optional list of known common causes. If None,
                                   common causes will be inferred from numeric columns.
        """
        self.sales_data_raw = sales_data.copy()  # Store copies to avoid modifying originals
        self.product_data_raw = product_data.copy() if product_data is not None else None
        self.store_data_raw = store_data.copy() if store_data is not None else None
        self.promotion_data_raw = promotion_data.copy() if promotion_data is not None else None

        self.treatment = treatment
        self.outcome = outcome
        # Store the user-provided default common causes separately
        self._user_default_common_causes = default_common_causes

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # State variables to be populated
        self.analysis_data: pd.DataFrame | None = None
        self.causal_graph_str: str | None = None
        self.common_causes: list[str] | None = None
        self.fitted_models: dict[str, Any] = {}
        self.last_run_results: dict[str, Any] = {}

        # --- Initialization Steps ---
        self._initialize_data_and_graph()

    def _initialize_data_and_graph(self):
        """Prepares data and defines the causal graph upon initialization."""
        self.logger.info("Initializing PromotionCausalAnalyzer...")
        # 1. Prepare Analysis Data
        try:
            self.analysis_data = prepare_analysis_data(
                sales_data=self.sales_data_raw,
                product_data=self.product_data_raw,
                store_data=self.store_data_raw,
                promotion_data=self.promotion_data_raw,
            )
            self.logger.info("Data preparation successful.")
        except ValueError as e:
            self.logger.error(f"ERROR during data preparation: {e}")
            self.analysis_data = None
            self.logger.error("Initialization failed due to data preparation error.")
            return
        except Exception as e:
            self.logger.error(f"UNEXPECTED ERROR during data preparation: {e}", exc_info=True)
            self.analysis_data = None
            self.logger.error("Initialization failed due to unexpected data preparation error.")
            return

        # 2. Define Causal Graph (only if data prep succeeded)
        if self.analysis_data is None:
            self.logger.warning("Cannot define causal graph because data preparation failed.")
            return

        try:
            (
                graph_str,
                _,  # treatment name (already stored in self.treatment)
                _,  # outcome name (already stored in self.outcome)
                common_causes_found,
            ) = define_causal_graph(
                analysis_data=self.analysis_data,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self._user_default_common_causes,  # Pass user preference
            )
            self.causal_graph_str = graph_str
            self.common_causes = common_causes_found  # Store the actual common causes used
            self.logger.info("Causal graph definition successful.")
        except ValueError as e:
            self.logger.error(f"ERROR during causal graph definition: {e}")
            # Proceeding without a graph might be possible but limit certain analyses
            self.causal_graph_str = None
            self.common_causes = None  # Set to None if graph def fails
        except Exception as e:
            self.logger.error(f"UNEXPECTED ERROR during causal graph definition: {e}", exc_info=True)
            self.causal_graph_str = None
            self.common_causes = None

        self.logger.info("Initialization complete.")

    # --- Core Functionality Methods ---

    def _run_naive_estimator(self, results: dict[str, Any]):
        self.logger.info("Running Naive Estimator...")
        try:
            results["naive"] = estimate_naive_ate(data=self.analysis_data, treatment=self.treatment, outcome=self.outcome)
        except Exception as e:
            self.logger.error(f"  ERROR running naive estimator: {e}")
            results["naive"] = {"error": str(e)}

    def _run_regression_estimator(self, results: dict[str, Any], has_common_causes: bool):
        if has_common_causes:
            self.logger.info("Running Regression Adjustment...")
            try:
                results["regression"] = estimate_regression_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=self.common_causes,
                )
            except Exception as e:
                self.logger.error(f"  ERROR running regression estimator: {e}")
                results["regression"] = {"error": str(e)}
        else:
            self.logger.warning("Skipping Regression Adjustment (no common causes).")
            results["regression"] = {"error": "Common causes not available."}

    def _run_matching_estimator(self, results: dict[str, Any], has_common_causes: bool):
        if has_common_causes:
            self.logger.info("Running Propensity Score Matching...")
            try:
                results["matching"] = estimate_matching_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=self.common_causes,
                )
            except Exception as e:
                self.logger.error(f"  ERROR running matching estimator: {e}")
                results["matching"] = {"error": str(e)}
        else:
            self.logger.warning("Skipping Propensity Score Matching (no common causes).")
            results["matching"] = {"error": "Common causes not available."}

    def _run_dowhy_estimator(
        self,
        results: dict[str, Any],
        method_key: str,
        dowhy_method_name: str,
        has_graph: bool,
        has_common_causes: bool,
    ):
        if has_graph and has_common_causes:
            self.logger.info(f"Running DoWhy ({dowhy_method_name.split('.')[-1].replace('_', ' ').title()})...")
            try:
                ate = estimate_dowhy_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=self.common_causes,
                    graph_str=self.causal_graph_str,
                    method_name=dowhy_method_name,
                )
                results[method_key] = {"ate": ate} if ate is not None else {"error": "DoWhy estimation returned None"}
            except Exception as e:
                self.logger.error(f"  ERROR running {method_key} estimator: {e}")
                results[method_key] = {"error": str(e)}
        else:
            reason = "causal graph" if not has_graph else "common causes"
            self.logger.warning(f"Skipping DoWhy ({dowhy_method_name.split('.')[-1].replace('_', ' ').title()}) (no {reason}).")
            results[method_key] = {"error": f"{reason.capitalize()} not available."}

    def _run_causalforest_estimator(self, results: dict[str, Any], has_common_causes: bool):
        if has_common_causes:
            self.logger.info("Running Causal Forest DML...")
            try:
                ate = estimate_causalforest_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=self.common_causes,
                )
                results["causal_forest"] = {"ate": ate} if ate is not None else {"error": "Causal Forest estimation returned None"}
            except Exception as e:
                self.logger.error(f"  ERROR running causal_forest estimator: {e}")
                results["causal_forest"] = {"error": str(e)}
        else:
            self.logger.warning("Skipping Causal Forest DML (no common causes).")
            results["causal_forest"] = {"error": "Common causes not available."}

    def _run_doubleml_estimator(
        self,
        results: dict[str, Any],
        method_key: str,
        ml_learner_name: str,
        has_common_causes: bool,
    ):
        if has_common_causes:
            self.logger.info(f"Running DoubleML IRM ({ml_learner_name})...")
            try:
                ate = estimate_doubleml_irm_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=self.common_causes,
                    ml_learner_name=ml_learner_name,
                )
                results[method_key] = {"ate": ate} if ate is not None else {"error": "DoubleML estimation returned None"}
            except Exception as e:
                self.logger.error(f"  ERROR running {method_key} estimator: {e}")
                results[method_key] = {"error": str(e)}
        else:
            self.logger.warning(f"Skipping DoubleML IRM ({ml_learner_name}) (no common causes).")
            results[method_key] = {"error": "Common causes not available."}

    def visualize_graph(self, save_path: str | None = None):
        """Visualizes the defined causal graph."""
        if self.causal_graph_str:
            try:
                visualize_causal_graph(self.causal_graph_str, save_path)
            except Exception as e:
                self.logger.error(f"Error during graph visualization call: {e}", exc_info=True)
        else:
            self.logger.error("Error: Causal graph is not available (was not defined or failed during initialization).")

    def run_all_analyses(self, run_estimators: list[str] | None = None) -> dict[str, Any]:
        """
        Runs a selection of causal analysis methods and returns the results.
        Delegates actual estimation to functions in causal_analysis.estimators.

        Args:
            run_estimators: A list of estimator keys to run. If None, runs a default set.
                            Supported keys: 'naive', 'regression', 'matching',
                            'dowhy_regression', 'dowhy_matching', 'causal_forest',
                            'doubleml_rf', 'doubleml_lasso'.

        Returns:
            dict: A dictionary containing the ATE estimates and results from different methods.
                  Results are also stored in `self.last_run_results`. Returns {'error': ...} on critical failure.
        """
        if self.analysis_data is None:
            msg = "Error: Analysis data not properly initialized. Cannot run analyses."
            self.logger.error(msg)
            return {"error": msg}

        results: dict[str, Any] = {}
        self.logger.info("\n--- Running Causal Analyses ---")

        default_set = [
            "naive",
            "regression",
            "matching",
            "causal_forest",
        ]  # Default estimators
        estimators_to_run = set(run_estimators if run_estimators is not None else default_set)

        # --- Prerequisites Check ---
        has_common_causes = self.common_causes is not None and len(self.common_causes) > 0
        has_graph = self.causal_graph_str is not None

        if not has_common_causes:
            self.logger.warning("Warning: Common causes not available. Estimators requiring confounders will be skipped or may fail.")
        if not has_graph:
            self.logger.warning("Warning: Causal graph not available. Graph-based estimators (DoWhy) will be skipped.")

        # --- Execute selected estimators by calling functions ---

        self._run_naive_estimator(results)

        if "regression" in estimators_to_run:
            self._run_regression_estimator(results, has_common_causes)

        if "matching" in estimators_to_run:
            self._run_matching_estimator(results, has_common_causes)

        if "dowhy_regression" in estimators_to_run:
            self._run_dowhy_estimator(
                results,
                "dowhy_regression",
                "backdoor.linear_regression",
                has_graph,
                has_common_causes,
            )

        if "dowhy_matching" in estimators_to_run:
            self._run_dowhy_estimator(
                results,
                "dowhy_matching",
                "backdoor.propensity_score_matching",
                has_graph,
                has_common_causes,
            )

        if "causal_forest" in estimators_to_run:
            self._run_causalforest_estimator(results, has_common_causes)

        if "doubleml_rf" in estimators_to_run:
            self._run_doubleml_estimator(results, "doubleml_rf", "RandomForest", has_common_causes)

        if "doubleml_lasso" in estimators_to_run:
            self._run_doubleml_estimator(results, "doubleml_lasso", "Lasso", has_common_causes)

        self.logger.info("\n--- Causal Analyses Complete ---")
        self.last_run_results = results  # Store results
        return results

    def fit_model_for_counterfactuals(self, model_key: str = "causal_forest", **kwargs) -> Any:
        """
        Fits a model suitable for counterfactuals and stores it.

        Args:
            model_key: Identifier for the model type to fit (default: 'causal_forest').
            **kwargs: Additional arguments passed to the model fitting function
                      (e.g., n_estimators for CausalForest).

        Returns:
            The fitted model object, or None if fitting fails.
        """
        if self.analysis_data is None:
            self.logger.error("Error: Analysis data not initialized. Cannot fit model.")
            return None
        if self.common_causes is None:
            # Check if common causes are strictly required for the model type
            if model_key in ["causal_forest"]:  # Add other types requiring confounders
                self.logger.error(f"Error: Common causes not available. Cannot fit '{model_key}' model.")
                return None
            else:
                self.logger.warning(f"Warning: Common causes not available, proceeding to fit '{model_key}' if possible without them.")

        model = None
        self.logger.info(f"\n--- Fitting Model for Counterfactuals ({model_key}) ---")
        if model_key == "causal_forest":
            try:
                # Ensure common_causes is a list, even if empty, for the function call
                fit_common_causes = self.common_causes if self.common_causes is not None else []
                model = fit_causal_forest_for_counterfactuals(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=fit_common_causes,
                    **kwargs,
                )
            except Exception as e:
                self.logger.error(f"  ERROR fitting {model_key} model: {e}", exc_info=True)
                model = None
        # Add elif blocks here for other model types ('regression', etc.) if needed
        # elif model_key == 'regression':
        #     # Fit and store a regression model (e.g., from statsmodels results)
        #     pass
        else:
            self.logger.warning(f"Model type '{model_key}' not currently supported for dedicated counterfactual fitting.")
            return None  # Explicitly return None if type not supported

        if model:
            self.fitted_models[model_key] = model
            self.logger.info(f"Model '{model_key}' fitted and stored.")
        else:
            self.logger.warning(f"Failed to fit model '{model_key}'.")
        return model

    def run_counterfactual_analysis(self, scenario: dict[str, Any], model_key: str = "causal_forest") -> dict[str, Any] | None:
        """
        Runs counterfactual analysis using a previously fitted model.
        Delegates actual simulation to causal_analysis.counterfactual.

        Args:
            scenario: Dictionary defining the counterfactual scenario (e.g., {'set_treatment': 1}).
            model_key: The key of the fitted model to use (default: 'causal_forest').

        Returns:
            Dictionary with counterfactual results, or None if simulation failed.
        """
        if self.analysis_data is None:
            self.logger.error("Error: Analysis data not initialized. Cannot run counterfactuals.")
            return None
        if self.common_causes is None:
            # Check if model type strictly needs common causes for prediction
            if model_key in ["causal_forest"]:
                self.logger.error(f"Error: Common causes not available, cannot run counterfactuals for '{model_key}'.")
                return None
            else:
                self.logger.warning(f"Warning: Common causes not available, proceeding with counterfactuals for '{model_key}' if possible.")

        model = self.fitted_models.get(model_key)
        if model is None:
            self.logger.error(f"Error: Model '{model_key}' not found or not fitted. Fit the model first using `fit_model_for_counterfactuals`.")
            return None

        self.logger.info(f"\n--- Running Counterfactual Analysis (Model: {model_key}, Scenario: {scenario}) ---")
        try:
            # Ensure common_causes is a list, even if empty
            sim_common_causes = self.common_causes if self.common_causes is not None else []
            results = simulate_counterfactuals(
                model=model,
                data=self.analysis_data,
                treatment=self.treatment,
                common_causes=sim_common_causes,
                scenario=scenario,
            )
            # Print results here for immediate feedback
            if results:
                self.logger.info("Counterfactual Simulation Results:")
                for key, val in results.items():
                    if isinstance(val, float):
                        self.logger.info(f"  {key.replace('_', ' ').title()}: {val:.4f}")
                    else:
                        self.logger.info(f"  {key.replace('_', ' ').title()}: {val}")
            else:
                self.logger.warning("Counterfactual simulation returned None.")
            return results
        except Exception as e:
            self.logger.error(f"  ERROR during counterfactual simulation call: {e}", exc_info=True)
            return None

    def _get_ate_for_roi(self, ate_source: str, estimated_ate_override: float | None) -> float | None:
        """Determines the ATE to use for ROI calculation."""
        if estimated_ate_override is not None:
            self.logger.info(f"Using directly provided ATE={estimated_ate_override:.4f} for ROI calculation.")
            return float(estimated_ate_override)

        if not self.last_run_results:
            self.logger.error("Error: No previous analysis results found in `last_run_results` to get ATE for ROI.")
            return None

        source_result = self.last_run_results.get(ate_source)
        if source_result is None:
            self.logger.error(f"Error: Result for ATE source '{ate_source}' not found in last run results.")
            self.logger.info(f"Available results: {list(self.last_run_results.keys())}")
            return None

        final_ate: float | None = None
        if isinstance(source_result, dict):
            if "ate" in source_result and isinstance(source_result["ate"], (int, float)):
                final_ate = float(source_result["ate"])
            elif "naive_ate" in source_result and isinstance(source_result["naive_ate"], (int, float)):
                final_ate = float(source_result["naive_ate"])
            elif "error" in source_result:
                self.logger.error(f"Error: Cannot use ATE from '{ate_source}' due to previous error: {source_result['error']}")
                return None
        elif isinstance(source_result, (int, float)):
            final_ate = float(source_result)

        if final_ate is None:
            self.logger.error(f"Error: Could not extract a valid numeric ATE value from result for '{ate_source}': {source_result}")
            return None

        self.logger.info(f"Using ATE={final_ate:.4f} from '{ate_source}' for ROI calculation.")
        return final_ate

    def _calculate_num_treated_units_for_roi(self) -> int | None:
        """Calculates the number of treated units from analysis_data."""
        if self.analysis_data is None:  # Should be checked before calling this helper ideally
            self.logger.error("Cannot calculate num_treated_units: analysis_data is None.")
            return None
        try:
            if self.treatment not in self.analysis_data.columns:
                raise KeyError(f"Treatment column '{self.treatment}' not found.")
            return int(self.analysis_data[self.treatment].sum())
        except KeyError as e:
            self.logger.error(f"Error calculating num_treated_units: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected Error calculating num_treated_units: {e}", exc_info=True)
            return None

    def _calculate_avg_baseline_sales_for_roi(self) -> float | None:
        """Calculates average baseline sales from analysis_data."""
        if self.analysis_data is None:  # Should be checked before calling this helper
            self.logger.error("Cannot calculate avg_baseline_sales: analysis_data is None.")
            return None
        try:
            if self.treatment not in self.analysis_data.columns:
                raise KeyError(f"Treatment column '{self.treatment}' not found.")
            if self.outcome not in self.analysis_data.columns:
                raise KeyError(f"Outcome column '{self.outcome}' not found.")

            control_mask = self.analysis_data[self.treatment] == 0
            if control_mask.any():
                control_sales = self.analysis_data.loc[control_mask, self.outcome]
                avg_val = control_sales.dropna().mean()
                if pd.isna(avg_val):
                    self.logger.warning("Warning: Average baseline sales calculation resulted in NaN. Using 0.0")
                    return 0.0
                return float(avg_val)
            else:
                self.logger.warning("Warning: No control units found. Cannot calculate average baseline sales. Using 0.0")
                return 0.0
        except KeyError as e:
            self.logger.error(f"Error calculating average_baseline_sales: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected Error calculating average_baseline_sales: {e}",
                exc_info=True,
            )
            return None

    def run_roi_analysis(
        self,
        ate_source: str = "causal_forest",
        promotion_cost_per_instance: float = 0.0,
        margin_percent: float = 0.0,
        estimated_ate: float | None = None,
        average_baseline_sales: float | None = None,
        num_treated_units: int | None = None,
    ) -> dict[str, Any] | None:
        """
        Calculates and interprets the ROI based on an ATE estimate from a previous analysis run.
        Delegates calculation and interpretation to causal_analysis.roi.

        Args:
            ate_source: Key from `last_run_results` indicating which method's ATE to use
                        (e.g., 'causal_forest', 'regression', 'matching'). Default 'causal_forest'.
            promotion_cost_per_instance: Cost per promotion application.
            margin_percent: Profit margin on sales (e.g., 0.2 for 20%).
            estimated_ate: Directly provide an ATE value, overriding ate_source lookup.
            average_baseline_sales: Avg sales for non-treated units. Calculated if None.
            num_treated_units: Number of treated units. Calculated if None.

        Returns:
            Dictionary with ROI calculation details and interpretation string, or None if failed.
            Example: {'roi_calculation': {...}, 'interpretation': "..."}
        """
        if self.analysis_data is None:
            self.logger.error("Error: Analysis data not available for ROI calculation.")
            return None

        final_ate = self._get_ate_for_roi(ate_source, estimated_ate)
        if final_ate is None:
            return None  # Error already logged by helper

        calc_num_treated_units = num_treated_units
        if calc_num_treated_units is None:
            calc_num_treated_units = self._calculate_num_treated_units_for_roi()
            if calc_num_treated_units is None:
                return None  # Error logged by helper

        calc_average_baseline_sales = average_baseline_sales
        if calc_average_baseline_sales is None:
            calc_average_baseline_sales = self._calculate_avg_baseline_sales_for_roi()
            if calc_average_baseline_sales is None:
                return None  # Error logged by helper

        self.logger.info(f"\n--- Running ROI Analysis (ATE Source: {ate_source if estimated_ate is None else 'Directly Provided'}) ---")
        try:
            roi_results = calculate_promotion_roi(
                estimated_ate=final_ate,
                average_baseline_sales=calc_average_baseline_sales,
                num_treated_units=calc_num_treated_units,
                promotion_cost_per_instance=promotion_cost_per_instance,
                margin_percent=margin_percent,
                treatment_variable=self.treatment,
            )

            # roi_results could be None if ATE was None, or dict with error
            interpretation = interpret_causal_impact(roi_results)

            # Structure the return value consistently
            final_result = {
                "roi_calculation": roi_results,
                "interpretation": interpretation,
            }
            # Print interpretation here as well for convenience
            self.logger.info(f"ROI Interpretation: {interpretation}")
            return final_result

        except Exception as e:
            self.logger.error(
                f"  ERROR during ROI calculation/interpretation call: {e}",
                exc_info=True,
            )
            return {
                "error": f"ROI calculation failed: {e}",
                "roi_calculation": None,
                "interpretation": None,
            }

    # --- Helper Methods (Internal / Optional) ---
    # This helper can be moved to data_preparation if it becomes complex
    # or kept here if simple and specific to orchestration needs.
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Placeholder: Identify holidays (example helper). Needs refinement."""
        # Simple example: Define a few fixed holidays (replace with actual logic/library)
        # Consider using pandas holidays or external library for better accuracy
        try:
            # Ensure dates are datetime objects
            date_objects = pd.to_datetime(dates).dt.date
            holidays = pd.to_datetime(["2023-01-01", "2023-07-04", "2023-12-25"]).date  # Example dates
            return date_objects.isin(holidays)
        except Exception as e:
            self.logger.warning(f"Warning: Failed to check holidays - {e}. Returning False for all dates.")
            return pd.Series([False] * len(dates), index=dates.index)

    # --- Deprecated / Removed Methods ---
    # Methods like _prepare_analysis_data, _define_causal_graph, estimate_*,
    # simulate_*, calculate_*, interpret_* are now handled by imported functions.


# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy dataframes
    sales = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-01",
                    "2023-01-02",
                    "2023-01-02",
                    "2023-01-03",
                    "2023-01-03",
                    "2023-01-04",
                    "2023-01-04",
                ]
            ),
            "store_id": [1, 2, 1, 2, 1, 2, 1, 2],
            "product_id": [101, 101, 102, 101, 101, 102, 101, 102],
            "sales": [10, 12, 5, 8, 11, 6, 13, 7],
            "price": [
                1.0,
                1.0,
                2.0,
                1.0,
                1.0,
                2.0,
                1.0,
                2.0,
            ],  # Add a potential confounder
            "marketing_spend": [
                100,
                150,
                100,
                150,
                120,
                110,
                130,
                120,
            ],  # Another potential confounder
        }
    )
    promotions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-03"]),
            "store_id": [1, 1],  # Only store 1
            "product_id": [101, 101],  # Only product 101
            "promotion_applied": [
                1,
                1,
            ],  # Promotion applied on day 1 and 3 for product 101 at store 1
        }
    )

    print("--- Creating Analyzer Instance ---")
    # Example: Specify known common causes
    analyzer = PromotionCausalAnalyzer(
        sales_data=sales,
        promotion_data=promotions,
        # product_data=... , # Add if available
        # store_data=... , # Add if available
        treatment="promotion_applied",
        outcome="sales",
        default_common_causes=["price", "marketing_spend"],
    )

    # Check if initialization was successful before proceeding
    if analyzer.analysis_data is not None:
        print("\n--- Analysis Data Head ---")
        print(analyzer.analysis_data.head())

        print("\n--- Visualizing Graph ---")
        analyzer.visualize_graph()  # Display plot interactively (if environment supports it)
        # analyzer.visualize_graph(save_path="causal_graph.png") # Save to file

        print("\n--- Running Analyses (Default Set) ---")
        analysis_results = analyzer.run_all_analyses()  # Runs ['naive', 'regression', 'matching', 'causal_forest'] by default

        print("\n--- Analysis Results Summary ---")
        # Iterate through results and print relevant info (e.g., ATE)
        for method, result in analysis_results.items():
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  {method}: ERROR - {result['error']}")
                elif "ate" in result and result["ate"] is not None:
                    ate_val = result["ate"]
                    pval_info = f", p-value: {result['p_value']:.3f}" if "p_value" in result and result["p_value"] is not None else ""
                    print(f"  {method}: ATE = {ate_val:.4f}{pval_info}")
                elif "naive_ate" in result:  # Handle naive structure
                    print(f"  {method}: Naive ATE = {result['naive_ate']:.4f}")
                else:
                    # Handle cases where ATE might be None or dict has unexpected structure
                    print(f"  {method}: Result structure non-standard or ATE is None - {result}")
            elif isinstance(result, float):  # Should generally be dict now
                print(f"  {method}: ATE = {result:.4f}")
            else:
                print(f"  {method}: No result or unexpected format - {result}")

        # --- Example: Counterfactual Analysis ---
        # Requires optional dependency: econml
        if CausalForestDML is not None:
            print("\n--- Fitting Model for Counterfactuals (Causal Forest) ---")
            # Fit Causal Forest (using default parameters from function)
            cf_model = analyzer.fit_model_for_counterfactuals(
                model_key="causal_forest",
                # Pass specific kwargs for CausalForestDML if needed:
                # n_estimators=50,
                # min_samples_leaf=5
            )

            if cf_model:  # Check if model fitting was successful
                print("\n--- Running Counterfactual Scenario (Apply Promotion to All) ---")
                cf_results_treat_all = analyzer.run_counterfactual_analysis(scenario={"set_treatment": 1}, model_key="causal_forest")

                print("\n--- Running Counterfactual Scenario (Remove Promotion from All) ---")
                cf_results_treat_none = analyzer.run_counterfactual_analysis(scenario={"set_treatment": 0}, model_key="causal_forest")
            else:
                print("Skipping counterfactual analysis as Causal Forest model could not be fitted.")
        else:
            print("\nSkipping Counterfactual/Causal Forest examples (EconML not installed).")

        # --- Example: ROI Analysis ---
        print("\n--- Running ROI Analysis ---")
        # Use ATE from a reliable method (e.g., 'regression' or 'causal_forest' if run)
        roi_analysis_results = analyzer.run_roi_analysis(
            ate_source="regression",  # Choose which ATE result to base ROI on
            promotion_cost_per_instance=0.50,  # Example cost
            margin_percent=0.25,  # Example margin (25%)
        )

        # ROI results dictionary structure: {'roi_calculation': {...}, 'interpretation': "..."}
        # Interpretation is already printed by the function call, but can be accessed:
        if roi_analysis_results and roi_analysis_results.get("roi_calculation"):
            # print(f"ROI Calculation Details: {roi_analysis_results['roi_calculation']}")
            pass  # Already printed
        elif roi_analysis_results and roi_analysis_results.get("error"):
            print(f"ROI Analysis Error: {roi_analysis_results['error']}")
        else:
            print("ROI Analysis did not produce a valid result or interpretation.")

    else:
        print("Analyzer initialization failed, cannot proceed with examples.")
