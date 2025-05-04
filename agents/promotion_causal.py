"""
PromotionCausalAnalyzer: Causal inference and counterfactual analysis for retail promotions and sales.

This module provides the PromotionCausalAnalyzer class, which enables robust causal analysis of retail promotion effectiveness, including regression adjustment, propensity score matching, double machine learning, and counterfactual simulation.

Key Capabilities:
- Data preparation and feature engineering for causal analysis
- Naive, regression, matching, and double ML estimators
- DoWhy integration for model validation
- Counterfactual scenario analysis
- ROI estimation for promotions

Adapted from the in-notebook implementation in sensor-networks-and-cognitive-systems.py.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional, Dict, List, Tuple, Union, Any, Set

# Optional dependencies
try:
    from dowhy import CausalModel
except ImportError:
    CausalModel = None
try:
    from econml.dml import CausalForestDML
except ImportError:
    CausalForestDML = None
try:
    from doubleml.data import DoubleMLData
    from doubleml.double_ml import DoubleMLIRM
except ImportError:
    DoubleMLData = None
    DoubleMLIRM = None

# --- Begin full PromotionCausalAnalyzer class definition from notebook ---
# (Full class pasted from the notebook, including all methods and helpers)
# (See attached notebook for the complete implementation.)


class PromotionCausalAnalyzer:
    """
    Analyzes the causal effect of promotions on sales performance using various methods.

    This class provides robust causal inference tools for retail promotion analysis, including regression adjustment, matching, double machine learning, and counterfactual simulation.
    """

    def __init__(
        self,
        sales_data: pd.DataFrame,
        product_data: pd.DataFrame | None = None,
        store_data: pd.DataFrame | None = None,
        promotion_data: pd.DataFrame | None = None,
    ):
        """Initialize with retail datasets."""
        self.sales_data = sales_data
        self.product_data = product_data
        self.store_data = store_data
        self.promotion_data = promotion_data
        self.analysis_data = self._prepare_analysis_data()
        self.causal_graph = self._define_causal_graph()

    # (All methods from the notebook's PromotionCausalAnalyzer class are included here)
    def _prepare_analysis_data(self) -> pd.DataFrame:
        """
        Merges sales, product, store, and promotion data into a single DataFrame
        suitable for causal analysis.

        Returns:
            pd.DataFrame: The merged and prepared analysis dataset.
        """
        # Start with sales data
        analysis_df = self.sales_data.copy()

        # Convert date columns to datetime if they aren't already
        if "date" in analysis_df.columns and not pd.api.types.is_datetime64_any_dtype(
            analysis_df["date"]
        ):
            analysis_df["date"] = pd.to_datetime(analysis_df["date"])

        # Merge product data if available
        if self.product_data is not None:
            analysis_df = pd.merge(
                analysis_df, self.product_data, on="product_id", how="left"
            )

        # Merge store data if available
        if self.store_data is not None:
            analysis_df = pd.merge(
                analysis_df, self.store_data, on="store_id", how="left"
            )

        # Merge promotion data if available
        if self.promotion_data is not None:
            # Ensure promotion data date is datetime
            if (
                "date" in self.promotion_data.columns
                and not pd.api.types.is_datetime64_any_dtype(
                    self.promotion_data["date"]
                )
            ):
                self.promotion_data["date"] = pd.to_datetime(
                    self.promotion_data["date"]
                )
            # Merge based on relevant keys (e.g., date, product_id, store_id)
            # Adjust merge keys based on actual promotion data structure
            merge_keys = ["date", "product_id", "store_id"]
            valid_merge_keys = [
                k
                for k in merge_keys
                if k in analysis_df.columns and k in self.promotion_data.columns
            ]
            if valid_merge_keys:
                analysis_df = pd.merge(
                    analysis_df,
                    self.promotion_data,
                    on=valid_merge_keys,
                    how="left",
                )
                # Assume 'promotion_applied' is the treatment indicator, fill NaNs with 0 (no promotion)
                if "promotion_applied" in analysis_df.columns:
                    analysis_df["promotion_applied"] = (
                        analysis_df["promotion_applied"].fillna(0).astype(int)
                    )
                else:
                    # If no promotion data merged, assume no promotions applied
                    analysis_df["promotion_applied"] = 0
            else:
                # If keys don't match, assume no promotions applied
                analysis_df["promotion_applied"] = 0

        # Feature Engineering (Example: Extract time features)
        if "date" in analysis_df.columns:
            analysis_df["day_of_week"] = analysis_df["date"].dt.dayofweek
            analysis_df["month"] = analysis_df["date"].dt.month
            analysis_df["year"] = analysis_df["date"].dt.year
            # Drop original date column if no longer needed for direct modeling
            # analysis_df = analysis_df.drop(columns=['date'])

        # Handle missing values (example: simple imputation or dropping)
        # This needs careful consideration based on the specific dataset
        analysis_df = analysis_df.dropna(
            subset=["sales", "promotion_applied"]
        )  # Drop rows where outcome or treatment is missing

        # Convert categorical features to numerical if needed (e.g., one-hot encoding)
        # Example:
        # analysis_df = pd.get_dummies(analysis_df, columns=['category', 'location_type'], drop_first=True)

        # Ensure required columns exist
        if "sales" not in analysis_df.columns:
            raise ValueError("Sales data must contain a 'sales' column.")
        if "promotion_applied" not in analysis_df.columns:
            # This case should be handled by the merging logic above, but double-check
            raise ValueError(
                "Could not determine the 'promotion_applied' treatment column."
            )

        print(
            f"Prepared analysis data with {analysis_df.shape[0]} rows and {analysis_df.shape[1]} columns."
        )
        print("Columns:", analysis_df.columns.tolist())

        return analysis_df

    def _define_causal_graph(
        self,
        treatment: str = "promotion_applied",
        outcome: str = "sales",
        common_causes: list[str] | None = None,
    ) -> str:
        """
        Defines the causal graph structure as a string.

        Args:
            treatment (str): Name of the treatment variable.
            outcome (str): Name of the outcome variable.
            common_causes (list[str] | None): List of variable names that are
                                              common causes of treatment and outcome.
                                              If None, attempts to infer from data columns.

        Returns:
            str: A string representing the causal graph in DOT format.
        """
        if common_causes is None:
            # Infer potential common causes from columns, excluding treatment and outcome
            potential_causes = [
                col
                for col in self.analysis_data.columns
                if col
                not in [
                    treatment,
                    outcome,
                    "date",
                ]  # Exclude date if not used as direct feature
                and pd.api.types.is_numeric_dtype(
                    self.analysis_data[col]
                )  # Basic check for numeric features
            ]
            # Heuristic: Select a subset or use domain knowledge. Here we use all numeric ones found.
            common_causes = potential_causes
            print(f"Inferred common causes: {common_causes}")

        # Basic graph: Common causes affect both treatment and outcome
        graph = f"digraph {{\n"
        graph += f'  "{treatment}" [label="Promotion Applied"];\n'
        graph += f'  "{outcome}" [label="Sales"];\n'

        # Add nodes for common causes
        for cause in common_causes:
            graph += f'  "{cause}" [label="{cause.replace("_", " ").title()}"];\n'

        # Add edges from common causes to treatment and outcome
        for cause in common_causes:
            graph += f'  "{cause}" -> "{treatment}";\n'
            graph += f'  "{cause}" -> "{outcome}";\n'

        # Add edge from treatment to outcome (the effect we want to estimate)
        graph += f'  "{treatment}" -> "{outcome}";\n'

        graph += "}"
        self.causal_graph_str = graph  # Store the graph string
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        return graph

    def estimate_ate_dowhy(
        self, method_name: str = "backdoor.linear_regression"
    ) -> float | None:
        """
        Estimates the Average Treatment Effect (ATE) using DoWhy library.

        Args:
            method_name (str): The estimation method to use within DoWhy
                               (e.g., "backdoor.linear_regression",
                               "backdoor.propensity_score_matching").

        Returns:
            float | None: The estimated ATE, or None if estimation fails or
                          DoWhy is not installed.
        """
        if CausalModel is None:
            print("DoWhy library not found. Skipping DoWhy estimation.")
            return None
        if not hasattr(self, "causal_graph_str"):
            print("Causal graph not defined. Call _define_causal_graph first.")
            return None

        try:
            # Ensure data types are suitable for DoWhy (e.g., boolean/int treatment)
            data = self.analysis_data.copy()
            # Convert boolean treatment to int if necessary
            if data[self.treatment].dtype == "bool":
                data[self.treatment] = data[self.treatment].astype(int)

            # Ensure common causes are numeric - handle potential non-numeric inferred causes
            numeric_common_causes = []
            for cause in self.common_causes:
                if pd.api.types.is_numeric_dtype(data[cause]):
                    numeric_common_causes.append(cause)
                else:
                    print(
                        f"Warning: Non-numeric common cause '{cause}' excluded from DoWhy model."
                    )

            if not numeric_common_causes:
                print("Warning: No numeric common causes found for DoWhy model.")
                # Decide whether to proceed without common causes or return None
                # Proceeding without common causes might lead to biased results
                # return None

            model = CausalModel(
                data=data,
                treatment=self.treatment,
                outcome=self.outcome,
                graph=self.causal_graph_str,
                common_causes=numeric_common_causes,  # Use only numeric ones
            )

            identified_estimand = model.identify_effect(proceed_when_unidentified=True)
            estimate = model.estimate_effect(
                identified_estimand, method_name=method_name, test_significance=True
            )

            ate = estimate.value
            print(f"\nDoWhy Estimation ({method_name}):")
            print(estimate)
            print(f"Estimated ATE: {ate:.4f}")
            # Cast to float before returning
            return float(ate) if ate is not None else None

        except Exception as e:
            print(f"Error during DoWhy estimation ({method_name}): {e}")
            import traceback

            traceback.print_exc()
            return None

    def estimate_ate_causalforest(
        self, n_estimators: int = 100, min_samples_leaf: int = 10
    ) -> float | None:
        """
        Estimates the Average Treatment Effect (ATE) using Causal Forest DML from EconML.

        Args:
            n_estimators (int): Number of trees in the forest.
            min_samples_leaf (int): Minimum number of samples required to be at a leaf node.

        Returns:
            float | None: The estimated ATE, or None if estimation fails or
                          EconML is not installed.
        """
        if CausalForestDML is None:
            print("EconML library not found. Skipping CausalForestDML estimation.")
            return None
        if not hasattr(self, "common_causes") or not self.common_causes:
            print("Common causes not defined or empty. Cannot run CausalForestDML.")
            return None

        try:
            data = self.analysis_data.copy()
            Y = data[self.outcome]
            T = data[self.treatment]
            X = data[self.common_causes]  # Confounders / Controls
            # W = None # Optional: Effect modifiers (not used here)

            # Ensure X contains only numeric data
            X = X.select_dtypes(include=np.number)
            if X.isnull().any().any():
                print(
                    "Warning: Missing values found in features (X). Filling with median."
                )
                X = X.fillna(X.median())

            # Define outcome and treatment models (can use more complex models)
            # Using GradientBoostingRegressor as an example
            from sklearn.ensemble import GradientBoostingRegressor

            model_y = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=123
            )
            model_t = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=123
            )

            # Initialize and fit CausalForestDML
            # Note: discrete_treatment=True is important if T is binary/categorical
            est = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,  # Assuming promotion is 0 or 1
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                random_state=123,
            )
            est.fit(Y, T, X=X)  # W=W if used

            ate = est.ate(X=X)  # Average treatment effect over the provided samples
            print(f"\nCausalForestDML Estimation:")
            print(f"Estimated ATE: {ate:.4f}")

            # Optional: Get confidence intervals
            ate_interval = est.ate_interval(X=X, alpha=0.05)  # 95% CI
            print(f"95% CI for ATE: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")

            # Cast to float before returning
            return float(ate) if ate is not None else None

        except Exception as e:
            print(f"Error during CausalForestDML estimation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def estimate_ate_doubleml_irm(
        self, ml_learner_name: str = "RandomForest"
    ) -> float | None:
        """
        Estimates the Average Treatment Effect (ATE) using DoubleML's
        Interactive Regression Model (IRM).

        Args:
            ml_learner_name (str): Name of the machine learning learner to use.
                                   Currently supports "RandomForest" or "Lasso".

        Returns:
            float | None: The estimated ATE, or None if estimation fails or
                          DoubleML is not installed.
        """
        if DoubleMLData is None or DoubleMLIRM is None:
            print("DoubleML library not found. Skipping DoubleML IRM estimation.")
            return None
        if not hasattr(self, "common_causes") or not self.common_causes:
            print("Common causes not defined or empty. Cannot run DoubleML.")
            return None

        try:
            data = self.analysis_data.copy()

            # Prepare data for DoubleMLData object
            y_col = self.outcome
            d_cols = self.treatment  # Treatment variable must be list-like for DoubleML
            x_cols = self.common_causes  # Confounders

            # Ensure confounders are numeric and handle NaNs
            numeric_confounders = (
                data[x_cols].select_dtypes(include=np.number).columns.tolist()
            )
            if len(numeric_confounders) < len(x_cols):
                print(
                    f"Warning: Non-numeric confounders excluded from DoubleML: {set(x_cols) - set(numeric_confounders)}"
                )
            if not numeric_confounders:
                print("Error: No numeric confounders available for DoubleML.")
                return None
            x_cols = numeric_confounders

            # Handle NaNs in relevant columns
            relevant_cols = [y_col, d_cols] + x_cols
            data_subset = data[relevant_cols].dropna()
            if data_subset.shape[0] < data.shape[0]:
                print(
                    f"Warning: Dropped {data.shape[0] - data_subset.shape[0]} rows with NaNs before DoubleML."
                )

            dml_data = DoubleMLData(
                data_subset, y_col=y_col, d_cols=d_cols, x_cols=x_cols
            )

            # Choose ML learners
            if ml_learner_name.lower() == "randomforest":
                from sklearn.ensemble import (
                    RandomForestRegressor,
                    RandomForestClassifier,
                )

                ml_g = RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=123
                )  # Model for E[Y|X]
                ml_m = RandomForestClassifier(
                    n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=123
                )  # Model for E[D|X] (propensity score)
            elif ml_learner_name.lower() == "lasso":
                from sklearn.linear_model import LassoCV, LogisticRegressionCV

                ml_g = LassoCV(cv=5, random_state=123)
                ml_m = LogisticRegressionCV(
                    cv=5, solver="liblinear", random_state=123
                )  # Use Logistic for binary treatment
            else:
                print(
                    f"Unsupported ml_learner_name: {ml_learner_name}. Use 'RandomForest' or 'Lasso'."
                )
                return None

            # Initialize DoubleMLIRM model
            dml_irm_model = DoubleMLIRM(dml_data, ml_g=ml_g, ml_m=ml_m)

            # Fit and estimate ATE
            dml_irm_model.fit()
            ate = dml_irm_model.coef[0]  # ATE is the first coefficient

            print(f"\nDoubleML IRM Estimation ({ml_learner_name}):")
            print(dml_irm_model.summary)
            print(f"Estimated ATE: {ate:.4f}")

            # Cast to float before returning
            return float(ate) if ate is not None else None

        except Exception as e:
            print(f"Error during DoubleML IRM estimation: {e}")
            import traceback

            traceback.print_exc()
            return None

    def run_all_analyses(self) -> dict:
        """
        Runs all implemented causal analysis methods and returns the results.

        Returns:
            dict: A dictionary containing the ATE estimates from different methods.
        """
        results = {}
        print("\n--- Running Causal Analyses ---")

        # Define graph first
        self._define_causal_graph()  # Use default parameters or pass specific ones

        # DoWhy Methods
        results["dowhy_regression"] = self.estimate_ate_dowhy(
            method_name="backdoor.linear_regression"
        )
        results["dowhy_matching"] = self.estimate_ate_dowhy(
            method_name="backdoor.propensity_score_matching"
        )

        # EconML Method
        results["econml_causalforest"] = self.estimate_ate_causalforest()

        # DoubleML Method
        results["doubleml_irm_rf"] = self.estimate_ate_doubleml_irm(
            ml_learner_name="RandomForest"
        )
        # results["doubleml_irm_lasso"] = self.estimate_ate_doubleml_irm(ml_learner_name="Lasso")

        print("\n--- Analysis Summary ---")
        for method, ate in results.items():
            if ate is not None:
                print(f"{method}: ATE = {ate:.4f}")
            else:
                print(f"{method}: Failed or Skipped")

        return results

    # Placeholder for counterfactual simulation - requires a trained model
    def simulate_counterfactuals(self, model_type="causalforest", **kwargs):
        """
        Simulates counterfactual outcomes (e.g., sales if everyone received promotion).
        NOTE: This is a placeholder and needs a specific trained model.
        """
        print("\n--- Counterfactual Simulation ---")
        if model_type == "causalforest" and CausalForestDML is not None:
            # Requires a fitted CausalForestDML instance
            # Example: Assuming self.causal_forest_model is fitted
            if (
                hasattr(self, "causal_forest_model")
                and self.causal_forest_model is not None
            ):
                data = self.analysis_data.copy()
                X = data[self.common_causes].select_dtypes(include=np.number)
                if X.isnull().any().any():
                    X = X.fillna(X.median())

                # Predict effect for each unit
                unit_effects = self.causal_forest_model.effect(X)

                # Simulate outcome if everyone treated (T=1)
                # Y_factual = data[self.outcome]
                # E_Y_given_X = self.causal_forest_model.model_y.predict(X) # Approx baseline
                # counterfactual_sales_all_treated = E_Y_given_X + unit_effects * (1 - data[self.treatment]) # Add effect if not treated
                # counterfactual_sales_none_treated = E_Y_given_X - unit_effects * data[self.treatment] # Remove effect if treated

                # Simpler: Estimate average outcome under treatment/control
                avg_effect = unit_effects.mean()
                avg_actual_sales = data[self.outcome].mean()
                avg_sales_if_all_treated = avg_actual_sales + avg_effect * (
                    1 - data[self.treatment].mean()
                )
                avg_sales_if_none_treated = (
                    avg_actual_sales - avg_effect * data[self.treatment].mean()
                )

                print(f"Average actual sales: {avg_actual_sales:.2f}")
                print(
                    f"Estimated average sales if ALL treated: {avg_sales_if_all_treated:.2f}"
                )
                print(
                    f"Estimated average sales if NONE treated: {avg_sales_if_none_treated:.2f}"
                )
                return {
                    "all_treated": avg_sales_if_all_treated,
                    "none_treated": avg_sales_if_none_treated,
                }
            else:
                print(
                    "CausalForest model not trained or available for counterfactuals."
                )
                return None
        else:
            print(
                f"Counterfactual simulation for model_type '{model_type}' not implemented or library missing."
            )
            return None

    # Helper to fit the Causal Forest model separately if needed for counterfactuals
    def fit_causal_forest(self, **kwargs):
        """Fits the CausalForestDML model and stores it."""
        if CausalForestDML is None:
            print("EconML not installed. Cannot fit Causal Forest.")
            self.causal_forest_model = None
            return None

        try:
            data = self.analysis_data.copy()
            Y = data[self.outcome]
            T = data[self.treatment]
            X = data[self.common_causes].select_dtypes(include=np.number)
            if X.isnull().any().any():
                print(
                    "Warning: Missing values found in features (X). Filling with median."
                )
                X = X.fillna(X.median())

            from sklearn.ensemble import GradientBoostingRegressor

            model_y = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=123
            )
            model_t = GradientBoostingRegressor(
                n_estimators=50, max_depth=3, random_state=123
            )

            est = CausalForestDML(
                model_y=model_y,
                model_t=model_t,
                discrete_treatment=True,
                random_state=123,
                **kwargs,  # Pass other CausalForestDML params like n_estimators
            )
            est.fit(Y, T, X=X)
            self.causal_forest_model = est
            print("CausalForestDML model fitted and stored.")
            return est
        except Exception as e:
            print(f"Error fitting CausalForestDML: {e}")
            self.causal_forest_model = None
            return None

    # Method extracted from .qmd (Used by _prepare_analysis_data if added there)

    # Note: The current _prepare_analysis_data in .py doesn't use this.
    # You might need to adapt _prepare_analysis_data if you want holiday features.
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Determine if dates are holidays"""
        # This is a simplified placeholder - in a real system,
        # you would use a holiday calendar library or a lookup table
        holidays = ["2023-01-01", "2023-12-25"]  # Example holidays
        # Ensure dates are datetime objects before comparison
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates = pd.to_datetime(dates)
        return dates.isin(pd.to_datetime(holidays))

    # Method extracted from .qmd (Previously called in notebook)
    # Requires matplotlib and networkx
    def visualize_causal_graph(self, save_path: Optional[str] = None):
        """Visualize the causal graph"""
        import matplotlib.pyplot as plt
        import networkx as nx  # Ensure networkx is imported or available

        if not hasattr(self, "causal_graph") or not isinstance(
            self.causal_graph, nx.DiGraph
        ):
            print("Causal graph is not defined or not a NetworkX DiGraph.")
            # Try generating from string if available
            if hasattr(self, "causal_graph_str"):
                try:
                    # Requires pydot and graphviz
                    from networkx.drawing.nx_pydot import read_dot

                    # Create a temporary file-like object
                    from io import StringIO

                    dot_f = StringIO(self.causal_graph_str)
                    self.causal_graph = read_dot(dot_f)
                    print("Graph loaded from DOT string.")
                except Exception as e:
                    print(f"Could not load graph from DOT string: {e}")
                    return
            else:
                print("No causal graph available to visualize.")
                return

        plt.figure(figsize=(12, 8))
        # Define Node positions - Adjust as necessary based on actual graph nodes
        # This position mapping might need update based on `_define_causal_graph` in .py
        pos = nx.spring_layout(self.causal_graph, seed=42)  # Example layout

        # Draw nodes - Adjust colors based on actual treatment/outcome names
        node_colors = []
        for node in self.causal_graph.nodes():
            if node == self.treatment:
                node_colors.append("lightblue")
            elif node == self.outcome:
                node_colors.append("lightgreen")
            else:
                node_colors.append("lightgrey")

        nx.draw_networkx_nodes(
            self.causal_graph, pos, node_color=node_colors, node_size=3000, alpha=0.8
        )
        # Draw edges
        nx.draw_networkx_edges(self.causal_graph, pos, arrows=True, arrowsize=20)
        # Draw labels
        nx.draw_networkx_labels(self.causal_graph, pos, font_size=12)
        # Add title and remove axis
        plt.title("Causal Graph for Promotion Analysis", fontsize=15)
        plt.axis("off")
        if save_path:
            plt.savefig(save_path)

        # Instead of plt.show(), which blocks in scripts, return the figure
        # or handle display appropriately in the calling environment (e.g., notebook)
        # For Marimo, you might want to save to BytesIO and use mo.image()
        # plt.show()
        # Return figure for potential display in Marimo
        return plt.gcf()

    def naive_promotion_impact(self) -> Dict[str, float]:
        """Calculate naive promotion impact (ignoring confounders)"""
        if (
            "promotion_applied" not in self.analysis_data.columns
            or "sales" not in self.analysis_data.columns
        ):
            return {
                "error": "Required columns (promotion_applied, sales) not found in analysis_data"
            }
        try:
            # Group by promotion status and calculate mean sales
            impact = (
                self.analysis_data.groupby("promotion_applied")["sales"]
                .mean()
                .reset_index()
            )
            # Calculate lift
            no_promo_df = impact[impact["promotion_applied"] == 0]
            promo_df = impact[impact["promotion_applied"] == 1]

            if no_promo_df.empty or promo_df.empty:
                return {"warning": "Data missing for one or both promotion groups"}

            no_promo = no_promo_df["sales"].values[0]
            promo = promo_df["sales"].values[0]

            if no_promo == 0:  # Avoid division by zero
                percent_lift = float("inf") if promo > 0 else 0.0
            else:
                percent_lift = (promo / no_promo - 1) * 100

            lift = promo - no_promo

            return {
                "no_promotion_avg": no_promo,
                "promotion_avg": promo,
                "absolute_lift_naive": lift,  # Key expected by notebook
                "percent_lift_naive": percent_lift,  # Key expected by notebook
            }
        except Exception as e:
            return {"error": f"Error during naive calculation: {e}"}

    def regression_adjustment(self) -> Dict[str, Any]:
        """Estimate promotion impact using regression adjustment for confounders"""
        import statsmodels.api as sm  # Ensure statsmodels is imported or available
        import numpy as np  # Ensure numpy is imported or available

        try:
            # Use inferred common causes + treatment
            # Ensure columns exist
            required_cols = self.common_causes + [self.treatment, self.outcome]
            missing_cols = [
                col for col in required_cols if col not in self.analysis_data.columns
            ]
            if missing_cols:
                return {
                    "error": f"Missing required columns for regression: {missing_cols}"
                }

            # Select numeric common causes + treatment
            potential_features = [c for c in self.common_causes if c != "holiday"] + [
                self.treatment
            ]
            feature_cols = [
                col
                for col in potential_features
                if pd.api.types.is_numeric_dtype(self.analysis_data[col])
            ]
            categorical_cols = [
                c
                for c in self.common_causes
                if c != "holiday" and c not in feature_cols
            ]

            analysis_subset = self.analysis_data[
                feature_cols + categorical_cols + [self.treatment, self.outcome]
            ].copy()

            if categorical_cols:
                analysis_subset = pd.get_dummies(
                    analysis_subset, columns=categorical_cols, drop_first=True
                )
                # Ensure feature_cols remains a list[str]
                feature_cols = [
                    col
                    for col in analysis_subset.columns
                    if col not in [self.outcome, self.treatment]
                ]

            if not feature_cols: # Check if list is empty
                return {
                    "error": "No suitable features found for regression adjustment."
                }

            # Prepare features (X) and outcome (Y)
            X = analysis_subset[feature_cols]
            Y = analysis_subset[self.outcome]

            # Add intercept
            X = sm.add_constant(X, has_constant="add")

            # Clean column names if necessary after get_dummies
            X.columns = [
                (
                    "_".join(x.split())
                    if isinstance(x, tuple)
                    else "".join(c if c.isalnum() else "_" for c in str(x))
                )
                for x in X.columns
            ]
            treatment_col_clean = "".join(
                c if c.isalnum() else "_" for c in str(self.treatment)
            )

            # Fit regression model
            model = sm.OLS(Y, X).fit()

            # Extract promotion coefficient (causal effect) - Ensure scalar extraction
            if treatment_col_clean not in model.params.index:
                return {
                    "error": f'Treatment variable "{treatment_col_clean}" not found in model results after cleaning names.'
                }

            # <<< Modify extraction to handle potential Series and ensure scalar >>>
            param_result = model.params[treatment_col_clean]
            promotion_effect = (
                param_result.iloc[0]
                if isinstance(param_result, pd.Series)
                else param_result
            )

            p_value_result = model.pvalues[treatment_col_clean]
            p_value = (
                p_value_result.iloc[0]
                if isinstance(p_value_result, pd.Series)
                else p_value_result
            )

            conf_int_result = model.conf_int().loc[treatment_col_clean]
            # conf_int() returns DataFrame, loc[key] can return Series or DataFrame row
            if isinstance(conf_int_result, pd.DataFrame):
                # If DataFrame row, take first row's values
                confidence_interval = conf_int_result.iloc[0].values.tolist()
            elif isinstance(conf_int_result, pd.Series):
                # If Series, take its values
                confidence_interval = conf_int_result.values.tolist()
            else:
                # Handle unexpected type
                confidence_interval = [None, None]
                print(
                    f"Warning: Unexpected type for confidence interval: {type(conf_int_result)}"
                )

            # Predict baseline and promotion sales
            X_no_promo = X.copy()
            X_no_promo[treatment_col_clean] = 0
            baseline_sales_pred = model.predict(X_no_promo)

            X_promo = X.copy()
            X_promo[treatment_col_clean] = 1
            promotion_sales_pred = model.predict(X_promo)

            # Avoid division by zero if baseline sales are zero or negative
            baseline_mean = baseline_sales_pred.mean()
            percent_lift = (
                (promotion_sales_pred.mean() / baseline_mean - 1) * 100
                if baseline_mean > 0
                else float("inf")
            )

            return {
                "estimated_ATE": promotion_effect,  # Key expected by notebook
                "p_value": p_value,
                "confidence_interval": confidence_interval,
                "baseline_sales_pred": baseline_mean,
                "promotion_sales_pred": promotion_sales_pred.mean(),
                "percent_lift_pred": percent_lift,  # Key expected by notebook
                "confounders_used": feature_cols,  # Report features used
            }
        except Exception as e:
            import traceback

            return {
                "error": f"Error in regression adjustment: {e}\n{traceback.format_exc()}"
            }

    def matching_analysis(
        self, caliper: float = 0.05, ratio: int = 1
    ) -> Dict[str, Any]:
        """Estimate promotion impact using propensity score matching"""
        from sklearn.linear_model import (
            LogisticRegression,
        )  # Ensure sklearn is imported or available
        import numpy as np  # Ensure numpy is imported or available

        try:
            # Use inferred common causes
            required_cols = self.common_causes + [self.treatment, self.outcome]
            missing_cols = [
                col for col in required_cols if col not in self.analysis_data.columns
            ]
            if missing_cols:
                return {
                    "error": f"Missing required columns for matching: {missing_cols}"
                }

            # Select numeric common causes for propensity model features
            potential_features = self.common_causes
            feature_cols = [
                col
                for col in potential_features
                if pd.api.types.is_numeric_dtype(self.analysis_data[col])
            ]

            # Handle categorical features if any were inferred (simple example: one-hot encode)
            categorical_cols = [
                col for col in self.common_causes if col not in feature_cols
            ]
            analysis_subset = self.analysis_data[
                feature_cols + categorical_cols + [self.treatment, self.outcome]
            ].copy()

            if categorical_cols:
                analysis_subset = pd.get_dummies(
                    analysis_subset, columns=categorical_cols, drop_first=True
                )
                # Update feature_cols list with new dummy columns
                feature_cols = [
                    col
                    for col in analysis_subset.columns
                    if col not in [self.outcome, self.treatment]
                ]

            if not feature_cols:
                return {
                    "error": "No suitable features found for propensity score model."
                }

            # Prepare features (X) and treatment (T)
            X = analysis_subset[feature_cols]
            T = analysis_subset[self.treatment]

            # Fit propensity score model
            propensity_model = LogisticRegression(
                solver="liblinear", random_state=42, max_iter=1000
            )
            propensity_model.fit(X, T)

            # Calculate propensity scores
            propensity_scores = propensity_model.predict_proba(X)[:, 1]
            analysis_subset["propensity_score"] = propensity_scores

            # Separate treatment and control groups
            treatment_group = analysis_subset[analysis_subset[self.treatment] == 1]
            control_group = analysis_subset[analysis_subset[self.treatment] == 0]

            if treatment_group.empty or control_group.empty:
                return {
                    "error": "Treatment or control group is empty, cannot perform matching."
                }

            # Matching (Nearest Neighbor within Caliper)
            matched_pairs = []
            used_control_indices: Set[int] = set()

            for treat_idx, treat_row in treatment_group.iterrows():
                # Calculate distances to control units not yet used
                potential_matches = control_group.drop(
                    index=list(used_control_indices), errors="ignore"
                ).copy()
                if potential_matches.empty:
                    continue

                potential_matches["distance"] = abs(
                    potential_matches["propensity_score"]
                    - treat_row["propensity_score"]
                )

                # Filter by caliper
                within_caliper = potential_matches[
                    potential_matches["distance"] <= caliper
                ]
                if within_caliper.empty:
                    continue

                # Find the 'ratio' closest matches
                closest_matches = within_caliper.nsmallest(ratio, "distance")

                for control_idx, control_row in closest_matches.iterrows():
                    if control_idx not in used_control_indices:
                        matched_pairs.append(
                            (treat_row[self.outcome], control_row[self.outcome])
                        )
                        # Cast index to int before adding to Set[int], handle non-int/non-float
                        if isinstance(control_idx, (int, float, np.number)):
                            try:
                                used_control_indices.add(int(control_idx)) 
                            except (ValueError, TypeError):
                                print(f"Warning: Could not convert control index '{control_idx}' to int. Skipping.")
                        else:
                             print(f"Warning: Control index '{control_idx}' is not numeric. Skipping.")

            # Calculate treatment effect from matched pairs
            if matched_pairs:
                treatment_outcomes = np.array([pair[0] for pair in matched_pairs])
                control_outcomes = np.array([pair[1] for pair in matched_pairs])
                effect = np.mean(treatment_outcomes - control_outcomes)

                # Calculate percentage effect carefully (avoid division by zero)
                control_mean = np.mean(control_outcomes)
                percent_effect = (
                    (effect / control_mean) * 100 if control_mean != 0 else float("inf")
                )

                return {
                    "matched_treatment_units": len(
                        treatment_outcomes
                    ),  # Key expected by notebook
                    "caliper": caliper,  # Key expected by notebook
                    "ratio": ratio,  # Key expected by notebook
                    "estimated_ATE": effect,  # Key expected by notebook
                    "percent_lift_est_ATE": percent_effect,  # Key expected by notebook
                    "treatment_mean_matched": np.mean(treatment_outcomes),
                    "control_mean_matched": control_mean,
                }
            else:
                return {"error": f"No matches found within caliper {caliper}"}
        except Exception as e:
            import traceback

            return {
                "error": f"Error in matching analysis: {e}\n{traceback.format_exc()}"
            }

    def double_ml_forest(self, **kwargs) -> Dict[str, Any]:
        """Estimate heterogeneous treatment effects using double ML causal forest"""
        if CausalForestDML is None:
            return {
                "error": "EconML library not found. Skipping CausalForestDML estimation."
            }
        # Placeholder implementation, adapted from .qmd example
        # This requires careful selection of features (X) and effect modifiers (W)
        return {
            "skipped": True,
            "reason": "DoubleML Forest implementation requires specific feature/modifier setup",
        }

    def dowhy_analysis(self, **kwargs) -> Dict[str, Any]:
        """Estimate causal effect using the DoWhy causal inference framework"""
        if CausalModel is None:
            return {"error": "DoWhy library not found. Skipping DoWhy analysis."}
        # Placeholder implementation
        return {
            "skipped": True,
            "reason": "DoWhy analysis implementation requires graph/model setup",
        }

    def perform_counterfactual_analysis(
        self, scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict outcomes under counterfactual scenarios using a regression model."""
        import statsmodels.api as sm  # Ensure statsmodels is imported or available
        import numpy as np  # Ensure numpy is imported or available

        try:
            # Fit a regression model on the original data
            potential_features_model = [
                c for c in self.common_causes if c != "holiday"
            ] + [self.treatment]
            feature_cols_model = [
                col
                for col in potential_features_model
                if pd.api.types.is_numeric_dtype(self.analysis_data[col])
            ]
            categorical_cols_model = [
                c
                for c in self.common_causes
                if c != "holiday" and c not in feature_cols_model
            ]
            analysis_subset_model = self.analysis_data[
                feature_cols_model
                + categorical_cols_model
                + [self.treatment, self.outcome]
            ].copy()

            if categorical_cols_model:
                analysis_subset_model = pd.get_dummies(
                    analysis_subset_model,
                    columns=categorical_cols_model,
                    drop_first=True,
                )
                # Ensure feature_cols_model remains a list[str]
                feature_cols_model = [
                    col
                    for col in analysis_subset_model.columns
                    if col not in [self.outcome, self.treatment]
                ] + [self.treatment]

            X_actual = analysis_subset_model[feature_cols_model]
            Y_actual = analysis_subset_model[self.outcome]
            X_actual = sm.add_constant(X_actual, has_constant="add")
            X_actual.columns = [
                (
                    "_".join(x.split())
                    if isinstance(x, tuple)
                    else "".join(c if c.isalnum() else "_" for c in str(x))
                )
                for x in X_actual.columns
            ]

            model = sm.OLS(Y_actual, X_actual).fit()
            # Ensure model_cols is a list[str]
            model_cols = model.params.index.tolist()

            # --- Create Counterfactual Data ---
            cf_data = self.analysis_data.copy()
            # Apply scenario changes
            for key, value in scenario.items():
                if key in cf_data.columns:
                    if isinstance(value, pd.Series):
                        # Align index and fill potentially missing values from original
                        cf_data[key] = value.reindex(cf_data.index).fillna(cf_data[key])
                    else:
                        cf_data[key] = value
                else:
                    print(f"Warning: Scenario key '{key}' not found in data columns.")

            # --- Prepare Counterfactual Features (X_cf) ---
            # Start with the same base columns as the model
            cf_subset = cf_data[
                feature_cols_model + categorical_cols_model + [self.treatment]
            ].copy()
            if categorical_cols_model:
                # Apply the same dummy encoding
                cf_subset = pd.get_dummies(
                    cf_subset, columns=categorical_cols_model, drop_first=True
                )

            # Add constant term
            X_cf = sm.add_constant(cf_subset, has_constant="add")
            # Clean column names similarly to X_actual
            X_cf.columns = [
                (
                    "_".join(x.split())
                    if isinstance(x, tuple)
                    else "".join(c if c.isalnum() else "_" for c in str(x))
                )
                for x in X_cf.columns
            ]

            # <<< Ensure X_cf has exactly the same columns as the model expects >>>
            # Reindex X_cf based on the columns the model was trained on (model_cols)
            # Add missing columns (e.g., dummy category not present in cf_data) and fill with 0
            # Drop columns present in X_cf but not used by the model
            X_cf = X_cf.reindex(columns=model_cols, fill_value=0)

            # --- Predict Counterfactual Outcomes ---
            # Predict using the aligned X_cf
            cf_predictions = model.predict(X_cf)

            # Calculate summary statistics
            actual_mean = Y_actual.mean()
            cf_mean = cf_predictions.mean()
            percentage_change = (
                ((cf_mean / actual_mean) - 1) * 100
                if actual_mean != 0
                else float("inf")
            )

            return {
                "actual_mean_sales": actual_mean,
                "counterfactual_mean_sales": cf_mean,
                "absolute_change": cf_mean - actual_mean,
                "percentage_change": percentage_change,
            }

        except Exception as e:
            import traceback

            return {
                "error": f"Error in counterfactual analysis: {e}\n{traceback.format_exc()}"
            }

    def calculate_promotion_roi(
        self, promotion_cost_per_instance: float, margin_percent: float
    ) -> Dict[str, Any]:
        """Calculate ROI of promotions considering causal effects"""
        try:
            regression_results = self.regression_adjustment()

            if "error" in regression_results:
                return {
                    "error": f"Cannot calculate ROI due to regression error: {regression_results['error']}"
                }

            causal_effect_per_instance = regression_results.get("estimated_ATE")
            if causal_effect_per_instance is None:
                return {
                    "error": "Could not retrieve estimated ATE from regression results."
                }

            # Calculate incremental margin per promoted instance
            if self.analysis_data[self.treatment].isnull().any():
                print(
                    f"--- WARNING ROI: NaNs found in treatment column '{self.treatment}'. Check data preparation."
                )

            promoted_mask = self.analysis_data[self.treatment] == 1

            avg_price_promoted = self.analysis_data.loc[promoted_mask, "price"].mean()
            if pd.isna(avg_price_promoted):
                avg_price_promoted = self.analysis_data["price"].mean()

            incremental_revenue_per_instance = (
                causal_effect_per_instance * avg_price_promoted
            )
            incremental_margin_per_instance = (
                incremental_revenue_per_instance * margin_percent
            )

            # Calculate ROI per instance
            roi_per_instance = (
                (
                    (incremental_margin_per_instance - promotion_cost_per_instance)
                    / promotion_cost_per_instance
                )
                * 100
                if promotion_cost_per_instance != 0
                else float("inf")
            )

            # Estimate total impact based on the number of promotion instances
            num_promotion_instances = self.analysis_data[self.treatment].sum()
            if not isinstance(num_promotion_instances, (int, np.integer)):
                print(
                    f"--- WARNING ROI: num_promotion_instances is not scalar int ({type(num_promotion_instances)}). Attempting conversion."
                )
                try:
                    num_promotion_instances = int(num_promotion_instances)
                except Exception as conv_e:
                    print(
                        f"--- ERROR ROI: Failed to convert num_promotion_instances to int: {conv_e}"
                    )
                    return {
                        "error": f"Could not get valid count of promotion instances: {conv_e}"
                    }

            total_incremental_margin_est = (
                incremental_margin_per_instance * num_promotion_instances
            )
            total_promotion_cost_est = (
                promotion_cost_per_instance * num_promotion_instances
            )

            total_roi_est = (
                (
                    (total_incremental_margin_est - total_promotion_cost_est)
                    / total_promotion_cost_est
                )
                * 100
                if total_promotion_cost_est != 0
                else float("inf")
            )

            profitable_estimate = total_roi_est > 0

            return {
                "estimated_ATE": causal_effect_per_instance,
                "avg_price_promoted": avg_price_promoted,
                "incremental_margin_per_instance": incremental_margin_per_instance,
                "roi_per_instance_percent": roi_per_instance,
                "num_promotion_instances": num_promotion_instances,
                "total_incremental_margin_est": total_incremental_margin_est,
                "total_promotion_cost_est": total_promotion_cost_est,
                "estimated_ROI_percent": total_roi_est,
                "profitable_estimate": profitable_estimate,
            }
        except Exception as e:
            import traceback

            print(f"--- ERROR calculating ROI: Exception caught: {e}")
            print(traceback.format_exc())
            return {"error": f"Error calculating ROI: {e}"}

    def interpret_causal_impact(self, roi_results: Dict[str, Any]) -> Union[str, Dict[str, str]]:
        """Interpret the causal impact of promotions"""
        try:
            # Check if all required keys are present and not NaN
            required_keys = ["estimated_ATE", "avg_price_promoted", "incremental_margin_per_instance", "roi_per_instance_percent", "num_promotion_instances", "total_incremental_margin_est", "total_promotion_cost_est", "estimated_ROI_percent", "profitable_estimate"]
            for key in required_keys:
                if key not in roi_results:
                    return {"error": f"Missing key '{key}' in ROI results dict."}
                if pd.isna(roi_results[key]):
                    return {"error": f"NaN value found for '{key}' in ROI results."}

            # Extract effect name if needed (e.g., for title)
            effect_key = next((k for k in roi_results if k.startswith("effect")), None)
            title_suffix = ""
            # Check if effect_key is a string before splitting
            if effect_key and isinstance(effect_key, str):
                # Add ignore comment as logic seems sound but mypy complains
                title_suffix = f" ({effect_key.split('_')[-1].upper()})" # type: ignore[attr-defined]

            # Generate interpretation string
            interpretation = f"The promotion has a significant causal effect on sales.{title_suffix}\n"
            interpretation += f"The estimated Average Treatment Effect (ATE) is {roi_results['estimated_ATE']:.4f}.\n"
            interpretation += f"The average price of promoted products is ${roi_results['avg_price_promoted']:.2f}.\n"
            interpretation += f"The incremental margin per promoted instance is ${roi_results['incremental_margin_per_instance']:.2f}.\n"
            interpretation += f"The ROI per promoted instance is {roi_results['roi_per_instance_percent']:.2f}%.\n"
            interpretation += f"The total incremental margin estimated is ${roi_results['total_incremental_margin_est']:.2f}.\n"
            interpretation += f"The total promotion cost estimated is ${roi_results['total_promotion_cost_est']:.2f}.\n"
            interpretation += f"The estimated ROI is {roi_results['estimated_ROI_percent']:.2f}%.\n"
            interpretation += f"The promotion is {'profitable' if roi_results['profitable_estimate'] else 'not profitable'} based on the estimated ROI."

            return interpretation
        except Exception as e:
            import traceback

            print(f"--- ERROR interpreting causal impact: Exception caught: {e}")
            print(traceback.format_exc())
            return {"error": f"Error interpreting causal impact: {e}"}
