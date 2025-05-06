# agents/promotion_causal.py

"""
PromotionCausalAnalyzer: Orchestrates causal inference and counterfactual analysis for retail promotions.

This module provides the PromotionCausalAnalyzer class, which uses specialized modules
for data preparation, graph definition, estimation, counterfactuals, and ROI calculation.
"""

import pandas as pd
from typing import Any, Dict, Optional, List
import traceback # Keep for potential error handling within orchestrator

# Import from the new causal_analysis package
# Ensure this relative import works based on how the project is structured/run.
# If running scripts from the root, might need 'from agents.causal_analysis import ...'
try:
    from .causal_analysis import (
        prepare_analysis_data,
        define_causal_graph,
        visualize_causal_graph,
        estimate_naive_ate,
        estimate_regression_ate,
        estimate_matching_ate,
        estimate_dowhy_ate,
        estimate_causalforest_ate,
        estimate_doubleml_irm_ate,
        fit_causal_forest_for_counterfactuals,
        simulate_counterfactuals,
        calculate_promotion_roi,
        interpret_causal_impact
    )
except ImportError:
     # Fallback for running script directly or different project structure
     print("Attempting absolute import for causal_analysis package...")
     from agents.causal_analysis import (
        prepare_analysis_data,
        define_causal_graph,
        visualize_causal_graph,
        estimate_naive_ate,
        estimate_regression_ate,
        estimate_matching_ate,
        estimate_dowhy_ate,
        estimate_causalforest_ate,
        estimate_doubleml_irm_ate,
        fit_causal_forest_for_counterfactuals,
        simulate_counterfactuals,
        calculate_promotion_roi,
        interpret_causal_impact
    )


# Optional dependencies needed by the orchestrator itself (e.g., storing fitted models)
# CausalForestDML is used for type hinting if available
try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None # Allow storing CausalForest if available, but don't require


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
        product_data: Optional[pd.DataFrame] = None,
        store_data: Optional[pd.DataFrame] = None,
        promotion_data: Optional[pd.DataFrame] = None,
        treatment: str = "promotion_applied",
        outcome: str = "sales",
        default_common_causes: Optional[List[str]] = None,
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
        self.sales_data_raw = sales_data.copy() # Store copies to avoid modifying originals
        self.product_data_raw = product_data.copy() if product_data is not None else None
        self.store_data_raw = store_data.copy() if store_data is not None else None
        self.promotion_data_raw = promotion_data.copy() if promotion_data is not None else None

        self.treatment = treatment
        self.outcome = outcome
        # Store the user-provided default common causes separately
        self._user_default_common_causes = default_common_causes

        # State variables to be populated
        self.analysis_data: Optional[pd.DataFrame] = None
        self.causal_graph_str: Optional[str] = None
        self.common_causes: Optional[List[str]] = None
        self.fitted_models: Dict[str, Any] = {}
        self.last_run_results: Dict[str, Any] = {}

        # --- Initialization Steps ---
        self._initialize_data_and_graph()


    def _initialize_data_and_graph(self):
        """Prepares data and defines the causal graph upon initialization."""
        print("Initializing PromotionCausalAnalyzer...")
        # 1. Prepare Analysis Data
        try:
            self.analysis_data = prepare_analysis_data(
                sales_data=self.sales_data_raw,
                product_data=self.product_data_raw,
                store_data=self.store_data_raw,
                promotion_data=self.promotion_data_raw,
            )
            print("Data preparation successful.")
        except ValueError as e:
            print(f"ERROR during data preparation: {e}")
            self.analysis_data = None
            # Stop initialization if data prep fails critically
            print("Initialization failed due to data preparation error.")
            return
        except Exception as e:
            print(f"UNEXPECTED ERROR during data preparation: {e}")
            traceback.print_exc()
            self.analysis_data = None
            print("Initialization failed due to unexpected data preparation error.")
            return

        # 2. Define Causal Graph (only if data prep succeeded)
        # Ensure analysis_data is not None before proceeding
        if self.analysis_data is None:
             print("Cannot define causal graph because data preparation failed.")
             return

        try:
            (
                graph_str,
                _, # treatment name (already stored in self.treatment)
                _, # outcome name (already stored in self.outcome)
                common_causes_found,
            ) = define_causal_graph(
                analysis_data=self.analysis_data,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self._user_default_common_causes, # Pass user preference
            )
            self.causal_graph_str = graph_str
            self.common_causes = common_causes_found # Store the actual common causes used
            print("Causal graph definition successful.")
        except ValueError as e:
            print(f"ERROR during causal graph definition: {e}")
            # Proceeding without a graph might be possible but limit certain analyses
            self.causal_graph_str = None
            self.common_causes = None # Set to None if graph def fails
        except Exception as e:
            print(f"UNEXPECTED ERROR during causal graph definition: {e}")
            traceback.print_exc()
            self.causal_graph_str = None
            self.common_causes = None

        print("Initialization complete.")

    # --- Core Functionality Methods ---

    def visualize_graph(self, save_path: Optional[str] = None):
        """Visualizes the defined causal graph."""
        if self.causal_graph_str:
            # Call the function from the graph module
            try:
                 visualize_causal_graph(self.causal_graph_str, save_path)
            except Exception as e:
                 print(f"Error during graph visualization call: {e}")
                 traceback.print_exc()
        else:
            print("Error: Causal graph is not available (was not defined or failed during initialization).")

    def run_all_analyses(self, run_estimators: Optional[List[str]] = None) -> Dict[str, Any]:
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
            print(msg)
            return {"error": msg}

        # Initialize results dict
        results: Dict[str, Any] = {}
        print("\n--- Running Causal Analyses ---")

        default_set = ['naive', 'regression', 'matching', 'causal_forest'] # Default estimators
        estimators_to_run = set(run_estimators if run_estimators is not None else default_set)

        # --- Prerequisites Check ---
        has_common_causes = self.common_causes is not None and len(self.common_causes) > 0
        has_graph = self.causal_graph_str is not None

        if not has_common_causes:
             print("Warning: Common causes not available. Estimators requiring confounders will be skipped or may fail.")
        if not has_graph:
             print("Warning: Causal graph not available. Graph-based estimators (DoWhy) will be skipped.")

        # --- Execute selected estimators by calling functions ---

        if 'naive' in estimators_to_run:
            print("Running Naive Estimator...")
            try:
                results["naive"] = estimate_naive_ate(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome
                )
            except Exception as e:
                print(f"  ERROR running naive estimator: {e}")
                results["naive"] = {"error": str(e)}

        # Methods requiring common causes
        if 'regression' in estimators_to_run:
            if has_common_causes:
                print("Running Regression Adjustment...")
                try:
                    results["regression"] = estimate_regression_ate(
                        data=self.analysis_data,
                        treatment=self.treatment,
                        outcome=self.outcome,
                        common_causes=self.common_causes
                    )
                except Exception as e:
                    print(f"  ERROR running regression estimator: {e}")
                    results["regression"] = {"error": str(e)}
            else:
                print("Skipping Regression Adjustment (no common causes).")
                results["regression"] = {"error": "Common causes not available."}

        if 'matching' in estimators_to_run:
            if has_common_causes:
                print("Running Propensity Score Matching...")
                try:
                    # Can add matching params (caliper, ratio) here if needed
                    results["matching"] = estimate_matching_ate(
                        data=self.analysis_data,
                        treatment=self.treatment,
                        outcome=self.outcome,
                        common_causes=self.common_causes,
                    )
                except Exception as e:
                    print(f"  ERROR running matching estimator: {e}")
                    results["matching"] = {"error": str(e)}
            else:
                print("Skipping Propensity Score Matching (no common causes).")
                results["matching"] = {"error": "Common causes not available."}

        # Methods requiring graph and common causes
        if 'dowhy_regression' in estimators_to_run:
            if has_graph and has_common_causes:
                 print("Running DoWhy (Linear Regression)...")
                 try:
                     # estimate_dowhy_ate returns Optional[float], handle potential None
                     ate = estimate_dowhy_ate(
                         data=self.analysis_data,
                         treatment=self.treatment,
                         outcome=self.outcome,
                         common_causes=self.common_causes,
                         graph_str=self.causal_graph_str,
                         method_name="backdoor.linear_regression"
                     )
                     results["dowhy_regression"] = {"ate": ate} if ate is not None else {"error": "DoWhy estimation returned None"}
                 except Exception as e:
                     print(f"  ERROR running dowhy_regression estimator: {e}")
                     results["dowhy_regression"] = {"error": str(e)}
            else:
                 reason = "causal graph" if not has_graph else "common causes"
                 print(f"Skipping DoWhy (Linear Regression) (no {reason}).")
                 results["dowhy_regression"] = {"error": f"{reason.capitalize()} not available."}

        if 'dowhy_matching' in estimators_to_run:
             if has_graph and has_common_causes:
                 print("Running DoWhy (Propensity Score Matching)...")
                 try:
                     ate = estimate_dowhy_ate(
                         data=self.analysis_data,
                         treatment=self.treatment,
                         outcome=self.outcome,
                         common_causes=self.common_causes,
                         graph_str=self.causal_graph_str,
                         method_name="backdoor.propensity_score_matching"
                     )
                     results["dowhy_matching"] = {"ate": ate} if ate is not None else {"error": "DoWhy estimation returned None"}
                 except Exception as e:
                     print(f"  ERROR running dowhy_matching estimator: {e}")
                     results["dowhy_matching"] = {"error": str(e)}
             else:
                 reason = "causal graph" if not has_graph else "common causes"
                 print(f"Skipping DoWhy (Propensity Score Matching) (no {reason}).")
                 results["dowhy_matching"] = {"error": f"{reason.capitalize()} not available."}


        # ML based methods requiring common causes
        if 'causal_forest' in estimators_to_run:
            if has_common_causes:
                print("Running Causal Forest DML...")
                try:
                    # Can add causal forest params (n_estimators, etc.) here if needed
                    ate = estimate_causalforest_ate(
                        data=self.analysis_data,
                        treatment=self.treatment,
                        outcome=self.outcome,
                        common_causes=self.common_causes
                    )
                    results["causal_forest"] = {"ate": ate} if ate is not None else {"error": "Causal Forest estimation returned None"}
                except Exception as e:
                    print(f"  ERROR running causal_forest estimator: {e}")
                    results["causal_forest"] = {"error": str(e)}
            else:
                 print("Skipping Causal Forest DML (no common causes).")
                 results["causal_forest"] = {"error": "Common causes not available."}

        if 'doubleml_rf' in estimators_to_run:
            if has_common_causes:
                 print("Running DoubleML IRM (Random Forest)...")
                 try:
                     ate = estimate_doubleml_irm_ate(
                         data=self.analysis_data,
                         treatment=self.treatment,
                         outcome=self.outcome,
                         common_causes=self.common_causes,
                         ml_learner_name="RandomForest"
                     )
                     results["doubleml_rf"] = {"ate": ate} if ate is not None else {"error": "DoubleML estimation returned None"}
                 except Exception as e:
                     print(f"  ERROR running doubleml_rf estimator: {e}")
                     results["doubleml_rf"] = {"error": str(e)}
            else:
                 print("Skipping DoubleML IRM (Random Forest) (no common causes).")
                 results["doubleml_rf"] = {"error": "Common causes not available."}


        if 'doubleml_lasso' in estimators_to_run:
             if has_common_causes:
                 print("Running DoubleML IRM (Lasso)...")
                 try:
                     ate = estimate_doubleml_irm_ate(
                         data=self.analysis_data,
                         treatment=self.treatment,
                         outcome=self.outcome,
                         common_causes=self.common_causes,
                         ml_learner_name="Lasso"
                     )
                     results["doubleml_lasso"] = {"ate": ate} if ate is not None else {"error": "DoubleML estimation returned None"}
                 except Exception as e:
                     print(f"  ERROR running doubleml_lasso estimator: {e}")
                     results["doubleml_lasso"] = {"error": str(e)}
             else:
                 print("Skipping DoubleML IRM (Lasso) (no common causes).")
                 results["doubleml_lasso"] = {"error": "Common causes not available."}


        print("\n--- Causal Analyses Complete ---")
        self.last_run_results = results # Store results
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
            print("Error: Analysis data not initialized. Cannot fit model.")
            return None
        if self.common_causes is None:
             # Check if common causes are strictly required for the model type
             if model_key in ["causal_forest"]: # Add other types requiring confounders
                 print(f"Error: Common causes not available. Cannot fit '{model_key}' model.")
                 return None
             else:
                  print(f"Warning: Common causes not available, proceeding to fit '{model_key}' if possible without them.")


        model = None
        print(f"\n--- Fitting Model for Counterfactuals ({model_key}) ---")
        if model_key == "causal_forest":
            try:
                # Ensure common_causes is a list, even if empty, for the function call
                fit_common_causes = self.common_causes if self.common_causes is not None else []
                model = fit_causal_forest_for_counterfactuals(
                    data=self.analysis_data,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=fit_common_causes,
                    **kwargs
                )
            except Exception as e:
                print(f"  ERROR fitting {model_key} model: {e}")
                traceback.print_exc()
                model = None
        # Add elif blocks here for other model types ('regression', etc.) if needed
        # elif model_key == 'regression':
        #     # Fit and store a regression model (e.g., from statsmodels results)
        #     pass
        else:
            print(f"Model type '{model_key}' not currently supported for dedicated counterfactual fitting.")
            return None # Explicitly return None if type not supported

        if model:
            self.fitted_models[model_key] = model
            print(f"Model '{model_key}' fitted and stored.")
        else:
             print(f"Failed to fit model '{model_key}'.")
        return model

    def run_counterfactual_analysis(
        self, scenario: Dict[str, Any], model_key: str = "causal_forest"
    ) -> Optional[Dict[str, Any]]:
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
            print("Error: Analysis data not initialized. Cannot run counterfactuals.")
            return None
        if self.common_causes is None:
             # Check if model type strictly needs common causes for prediction
             if model_key in ["causal_forest"]:
                 print(f"Error: Common causes not available, cannot run counterfactuals for '{model_key}'.")
                 return None
             else:
                  print(f"Warning: Common causes not available, proceeding with counterfactuals for '{model_key}' if possible.")


        model = self.fitted_models.get(model_key)
        if model is None:
            print(f"Error: Model '{model_key}' not found or not fitted. Fit the model first using `fit_model_for_counterfactuals`.")
            return None

        print(f"\n--- Running Counterfactual Analysis (Model: {model_key}, Scenario: {scenario}) ---")
        try:
            # Ensure common_causes is a list, even if empty
            sim_common_causes = self.common_causes if self.common_causes is not None else []
            results = simulate_counterfactuals(
                model=model,
                data=self.analysis_data,
                treatment=self.treatment,
                common_causes=sim_common_causes,
                scenario=scenario
            )
            # Print results here for immediate feedback
            if results:
                 print("Counterfactual Simulation Results:")
                 for key, val in results.items():
                      if isinstance(val, float):
                           print(f"  {key.replace('_',' ').title()}: {val:.4f}")
                      else:
                           print(f"  {key.replace('_',' ').title()}: {val}")
            else:
                 print("Counterfactual simulation returned None.")
            return results
        except Exception as e:
            print(f"  ERROR during counterfactual simulation call: {e}")
            traceback.print_exc()
            return None

    def run_roi_analysis(
        self,
        ate_source: str = "causal_forest", # Method whose ATE to use (e.g., 'causal_forest', 'regression')
        promotion_cost_per_instance: float = 0.0, # Default to 0 cost if not specified
        margin_percent: float = 0.0, # Default to 0 margin if not specified
        estimated_ate: Optional[float] = None, # Allow overriding ATE directly
        average_baseline_sales: Optional[float] = None,
        num_treated_units: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
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
            print("Error: Analysis data not available for ROI calculation.")
            return None

        # Determine the ATE to use
        final_ate: Optional[float] = estimated_ate # Use override if provided
        if final_ate is None:
            if not self.last_run_results:
                print(f"Error: No previous analysis results found in `last_run_results`. Run `run_all_analyses` first or provide `estimated_ate`.")
                return None
            source_result = self.last_run_results.get(ate_source)
            if source_result is None:
                 print(f"Error: Result for ATE source '{ate_source}' not found in last run results.")
                 print(f"Available results: {list(self.last_run_results.keys())}")
                 return None
            # Extract ATE, handling different result structures (dict vs float)
            if isinstance(source_result, dict):
                 if "ate" in source_result and isinstance(source_result["ate"], (int, float)):
                     final_ate = float(source_result["ate"])
                 elif "naive_ate" in source_result and isinstance(source_result["naive_ate"], (int, float)): # Handle naive structure
                      final_ate = float(source_result["naive_ate"])
                 elif "error" in source_result:
                      print(f"Error: Cannot use ATE from '{ate_source}' due to previous error: {source_result['error']}")
                      return None
            elif isinstance(source_result, (int, float)): # Handle cases where only ATE float might have been stored
                 final_ate = float(source_result)

            if final_ate is None:
                 print(f"Error: Could not extract a valid numeric ATE value from result for '{ate_source}': {source_result}")
                 return None
            print(f"Using ATE={final_ate:.4f} from '{ate_source}' for ROI calculation.")

        # Calculate baseline sales and treated units if not provided
        if num_treated_units is None:
             try:
                 # Ensure treatment column exists and is numeric/boolean
                 if self.treatment not in self.analysis_data.columns:
                      raise KeyError(f"Treatment column '{self.treatment}' not found.")
                 num_treated_units = int(self.analysis_data[self.treatment].sum())
             except KeyError as e:
                 print(f"Error calculating num_treated_units: {e}")
                 return None
             except Exception as e:
                 print(f"Unexpected Error calculating num_treated_units: {e}")
                 traceback.print_exc()
                 return None

        if average_baseline_sales is None:
             try:
                 if self.treatment not in self.analysis_data.columns:
                     raise KeyError(f"Treatment column '{self.treatment}' not found.")
                 if self.outcome not in self.analysis_data.columns:
                      raise KeyError(f"Outcome column '{self.outcome}' not found.")

                 control_mask = (self.analysis_data[self.treatment] == 0)
                 if control_mask.any():
                     control_sales = self.analysis_data.loc[control_mask, self.outcome]
                     # Handle potential NaNs in outcome for controls
                     avg_val = control_sales.dropna().mean()
                     if pd.isna(avg_val):
                          print("Warning: Average baseline sales calculation resulted in NaN (maybe all controls had NaN outcome?). Using 0.0")
                          average_baseline_sales = 0.0
                     else:
                          average_baseline_sales = float(avg_val)
                 else:
                     print("Warning: No control units found (treatment == 0). Cannot calculate average baseline sales. Using 0.0")
                     average_baseline_sales = 0.0
             except KeyError as e:
                 print(f"Error calculating average_baseline_sales: {e}")
                 return None
             except Exception as e:
                 print(f"Unexpected Error calculating average_baseline_sales: {e}")
                 traceback.print_exc()
                 return None

        print(f"\n--- Running ROI Analysis (ATE Source: {ate_source if estimated_ate is None else 'Directly Provided'}) ---")
        try:
            roi_results = calculate_promotion_roi(
                estimated_ate=final_ate, # Use the determined ATE
                average_baseline_sales=average_baseline_sales,
                num_treated_units=num_treated_units,
                promotion_cost_per_instance=promotion_cost_per_instance,
                margin_percent=margin_percent,
                treatment_variable=self.treatment,
            )

            # roi_results could be None if ATE was None, or dict with error
            interpretation = interpret_causal_impact(roi_results)

            # Structure the return value consistently
            final_result = {"roi_calculation": roi_results, "interpretation": interpretation}
            # Print interpretation here as well for convenience
            print(f"ROI Interpretation: {interpretation}")
            return final_result

        except Exception as e:
            print(f"  ERROR during ROI calculation/interpretation call: {e}")
            traceback.print_exc()
            return {"error": f"ROI calculation failed: {e}", "roi_calculation": None, "interpretation": None}


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
             holidays = pd.to_datetime(["2023-01-01", "2023-07-04", "2023-12-25"]).date # Example dates
             return date_objects.isin(holidays)
        except Exception as e:
             print(f"Warning: Failed to check holidays - {e}. Returning False for all dates.")
             return pd.Series([False] * len(dates), index=dates.index)


    # --- Deprecated / Removed Methods ---
    # Methods like _prepare_analysis_data, _define_causal_graph, estimate_*,
    # simulate_*, calculate_*, interpret_* are now handled by imported functions.


# --- Example Usage ---
if __name__ == '__main__':
    # Create dummy dataframes
    sales = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-04', '2023-01-04']),
        'store_id': [1, 2, 1, 2, 1, 2, 1, 2],
        'product_id': [101, 101, 102, 101, 101, 102, 101, 102],
        'sales': [10, 12, 5, 8, 11, 6, 13, 7],
        'price': [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0], # Add a potential confounder
        'marketing_spend': [100, 150, 100, 150, 120, 110, 130, 120] # Another potential confounder
    })
    promotions = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-01-03']),
        'store_id': [1, 1], # Only store 1
        'product_id': [101, 101], # Only product 101
        'promotion_applied': [1, 1] # Promotion applied on day 1 and 3 for product 101 at store 1
    })

    print("--- Creating Analyzer Instance ---")
    # Example: Specify known common causes
    analyzer = PromotionCausalAnalyzer(
        sales_data=sales,
        promotion_data=promotions,
        # product_data=... , # Add if available
        # store_data=... , # Add if available
        treatment="promotion_applied",
        outcome="sales",
        default_common_causes=["price", "marketing_spend"]
    )

    # Check if initialization was successful before proceeding
    if analyzer.analysis_data is not None:
        print("\n--- Analysis Data Head ---")
        print(analyzer.analysis_data.head())

        print("\n--- Visualizing Graph ---")
        analyzer.visualize_graph() # Display plot interactively (if environment supports it)
        # analyzer.visualize_graph(save_path="causal_graph.png") # Save to file

        print("\n--- Running Analyses (Default Set) ---")
        analysis_results = analyzer.run_all_analyses() # Runs ['naive', 'regression', 'matching', 'causal_forest'] by default

        print("\n--- Analysis Results Summary ---")
        # Iterate through results and print relevant info (e.g., ATE)
        for method, result in analysis_results.items():
            if isinstance(result, dict):
                if "error" in result:
                    print(f"  {method}: ERROR - {result['error']}")
                elif "ate" in result and result["ate"] is not None:
                    ate_val = result['ate']
                    pval_info = f", p-value: {result['p_value']:.3f}" if 'p_value' in result and result['p_value'] is not None else ""
                    print(f"  {method}: ATE = {ate_val:.4f}{pval_info}")
                elif "naive_ate" in result: # Handle naive structure
                    print(f"  {method}: Naive ATE = {result['naive_ate']:.4f}")
                else:
                     # Handle cases where ATE might be None or dict has unexpected structure
                     print(f"  {method}: Result structure non-standard or ATE is None - {result}")
            elif isinstance(result, float): # Should generally be dict now
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

            if cf_model: # Check if model fitting was successful
                print("\n--- Running Counterfactual Scenario (Apply Promotion to All) ---")
                cf_results_treat_all = analyzer.run_counterfactual_analysis(
                    scenario={'set_treatment': 1},
                    model_key="causal_forest"
                )

                print("\n--- Running Counterfactual Scenario (Remove Promotion from All) ---")
                cf_results_treat_none = analyzer.run_counterfactual_analysis(
                    scenario={'set_treatment': 0},
                    model_key="causal_forest"
                )
            else:
                print("Skipping counterfactual analysis as Causal Forest model could not be fitted.")
        else:
             print("\nSkipping Counterfactual/Causal Forest examples (EconML not installed).")


        # --- Example: ROI Analysis ---
        print("\n--- Running ROI Analysis ---")
        # Use ATE from a reliable method (e.g., 'regression' or 'causal_forest' if run)
        roi_analysis_results = analyzer.run_roi_analysis(
            ate_source='regression', # Choose which ATE result to base ROI on
            promotion_cost_per_instance=0.50, # Example cost
            margin_percent=0.25 # Example margin (25%)
        )

        # ROI results dictionary structure: {'roi_calculation': {...}, 'interpretation': "..."}
        # Interpretation is already printed by the function call, but can be accessed:
        if roi_analysis_results and roi_analysis_results.get("roi_calculation"):
            # print(f"ROI Calculation Details: {roi_analysis_results['roi_calculation']}")
            pass # Already printed
        elif roi_analysis_results and roi_analysis_results.get("error"):
             print(f"ROI Analysis Error: {roi_analysis_results['error']}")
        else:
            print("ROI Analysis did not produce a valid result or interpretation.")

    else:
        print("Analyzer initialization failed, cannot proceed with examples.")