# region book:promotion-causal-imports

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict

# endregion book:promotion-causal-imports


# region book:promotion-causal-schemas
class NaiveImpact(BaseModel):
    no_promo_sales: float
    promo_sales: float
    absolute_lift: float
    percent_lift: float


class RegressionAdjustment(BaseModel):
    promotion_effect: float
    p_value: float
    r_squared: float


class MatchingImpact(BaseModel):
    matched_pairs: int
    average_treatment_effect: float
    percent_effect: float


class SegmentEffects(BaseModel):
    by_category: dict[str, float]
    by_store_tier: dict[str, float]


class DoubleMLResult(BaseModel):
    average_treatment_effect: float
    heterogeneous_effects: SegmentEffects


class DoWhyResult(BaseModel):
    regression_effect: float
    matching_effect: float
    random_refutation: str
    placebo_refutation: str


class CounterfactualScenario(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    overrides: dict[str, pd.Series | float]


class CounterfactualSummary(BaseModel):
    mean_predicted_sales: float
    total_predicted_sales: float
    min_predicted_sales: float
    max_predicted_sales: float
    actual_total_sales: float
    percentage_change: float


class CounterfactualResult(BaseModel):
    summary: CounterfactualSummary | None = None
    error: str | None = None


class PromotionROI(BaseModel):
    incremental_revenue: float
    incremental_profit: float
    roi_percent: float


# endregion book:promotion-causal-schemas


# region book:promotion-causal-analyzer
class PromotionCausalAnalyzer:
    """Analyzes the causal effect of promotions on sales performance"""

    def __init__(
        self,
        sales_data: pd.DataFrame,
        product_data: pd.DataFrame,
        store_data: pd.DataFrame,
        promotion_data: pd.DataFrame,
    ):
        """Initialize with retail datasets"""
        self.sales_data = sales_data
        self.product_data = product_data
        self.store_data = store_data
        self.promotion_data = promotion_data
        # Prepare the analysis dataset
        self.analysis_data = self._prepare_analysis_data()
        # Define causal graph structure
        self.causal_graph = self._define_causal_graph()

    # endregion book:promotion-causal-analyzer

    # region book:promotion-causal-prepare-data-merge
    def _prepare_analysis_data(self) -> pd.DataFrame:
        """Combine and prepare data for causal analysis"""
        # Merge sales with product attributes
        df = pd.merge(
            self.sales_data,
            self.product_data.drop(
                columns=self.product_data.columns.intersection(self.sales_data.columns).difference(
                    ["product_id"]
                )
            ),
            on="product_id",
            how="left",
            validate="many_to_one",
        )
        # Add store characteristics
        df = pd.merge(
            df,
            self.store_data.drop(
                columns=self.store_data.columns.intersection(df.columns).difference(["store_id"])
            ),
            on="store_id",
            how="left",
            validate="many_to_one",
        )
        # endregion book:promotion-causal-prepare-data-merge

        # region book:promotion-causal-prepare-data-promo
        # Add promotion flags
        df["date"] = pd.to_datetime(df["date"])
        promotions = self.promotion_data.copy()
        promotions["date"] = pd.to_datetime(promotions["date"])
        if "on_promotion" in df:
            promotions = promotions.rename(columns={"on_promotion": "_promotion_treatment"})
        df = pd.merge(
            df,
            promotions,
            on=["product_id", "store_id", "date"],
            how="left",
            validate="many_to_one",
            suffixes=(False, False),
        )
        if "_promotion_treatment" in df:
            supplied = df.pop("_promotion_treatment")
            existing = df["on_promotion"]
            if (existing.notna() & supplied.notna() & (existing != supplied)).any():
                raise ValueError("Sales and promotion data disagree on the treatment indicator.")
            df["on_promotion"] = existing.combine_first(supplied)
        if "on_promotion" not in df:
            raise ValueError("Could not determine the promotion treatment indicator.")
        if not df["on_promotion"].dropna().isin([0, 1]).all():
            raise ValueError("The promotion treatment indicator must be binary.")
        # Fill missing promotion flags with False
        df["on_promotion"] = df["on_promotion"].fillna(False).astype(bool)
        # Create calendar features
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["holiday"] = self._is_holiday(df["date"])
        # endregion book:promotion-causal-prepare-data-promo

        # region book:promotion-causal-prepare-data-lags
        # Create lagged features
        df = df.sort_values(["store_id", "product_id", "date"], kind="stable").reset_index(drop=True)
        for lag in [1, 2, 3, 7, 14]:
            df[f"sales_lag_{lag}"] = df.groupby(["product_id", "store_id"])["sales_units"].shift(lag)
            df[f"on_promotion_lag_{lag}"] = (
                df.groupby(["product_id", "store_id"])["on_promotion"].shift(lag).astype(float)
            )
        # Fill missing values
        df = df.fillna(0)
        return df
        # endregion book:promotion-causal-prepare-data-lags

    # region book:promotion-causal-is-holiday
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Determine if dates are holidays"""
        # This is a simplified placeholder - in a real system,
        # you would use a holiday calendar library or a lookup table
        holidays = ["2023-01-01", "2023-12-25"]  # Example holidays
        return dates.isin(pd.to_datetime(holidays))

    # endregion book:promotion-causal-is-holiday

    # region book:promotion-causal-define-graph
    def _define_causal_graph(self) -> nx.DiGraph:
        """Define the directed acyclic graph of causal relationships"""
        G = nx.DiGraph()
        # Add nodes
        nodes = [
            "on_promotion",  # Treatment variable
            "sales_units",  # Outcome variable
            "price",  # Mediator
            "day_of_week",  # Confounder
            "month",  # Confounder
            "weekend",  # Confounder
            "holiday",  # Confounder
            "store_traffic",  # Confounder
            "product_category",  # Confounder
            "store_tier",  # Confounder
        ]
        G.add_nodes_from(nodes)
        # endregion book:promotion-causal-define-graph

        # region book:promotion-causal-define-graph-edges
        # Add edges (causal relationships)
        edges = [
            # Promotion affects sales directly and through price
            ("on_promotion", "price"),
            ("on_promotion", "sales_units"),
            ("price", "sales_units"),
            # Confounders affect both treatment and outcome
            ("day_of_week", "on_promotion"),
            ("day_of_week", "sales_units"),
            ("month", "on_promotion"),
            ("month", "sales_units"),
            ("weekend", "on_promotion"),
            ("weekend", "sales_units"),
            ("holiday", "on_promotion"),
            ("holiday", "sales_units"),
            ("store_traffic", "on_promotion"),
            ("store_traffic", "sales_units"),
            ("product_category", "on_promotion"),
            ("product_category", "sales_units"),
            ("store_tier", "on_promotion"),
            ("store_tier", "sales_units"),
        ]
        G.add_edges_from(edges)

        return G
        # endregion book:promotion-causal-define-graph-edges

    # region book:promotion-causal-visualize-graph
    def visualize_causal_graph(self, save_path: str | None = None):
        """Visualize the causal graph"""
        plt.figure(figsize=(12, 8))
        # Node positions
        pos = {
            "on_promotion": (0.5, 0.5),
            "sales_units": (0.8, 0.5),
            "price": (0.65, 0.6),
            "day_of_week": (0.3, 0.7),
            "month": (0.3, 0.6),
            "weekend": (0.3, 0.5),
            "holiday": (0.3, 0.4),
            "store_traffic": (0.3, 0.3),
            "product_category": (0.5, 0.3),
            "store_tier": (0.5, 0.2),
        }
        # endregion book:promotion-causal-visualize-graph

        # region book:promotion-causal-visualize-graph-draw
        # Draw nodes
        nx.draw_networkx_nodes(
            self.causal_graph,
            pos,
            node_color=[
                "lightblue"
                if node == "on_promotion"
                else "lightgreen"
                if node == "sales_units"
                else "lightgrey"
                for node in self.causal_graph.nodes
            ],
            node_size=3000,
        )
        # Draw edges
        nx.draw_networkx_edges(self.causal_graph, pos, arrows=True, arrowsize=20)
        # Draw labels
        nx.draw_networkx_labels(self.causal_graph, pos, font_size=10)
        plt.title("Causal Graph for Promotion Analysis")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        # endregion book:promotion-causal-visualize-graph-draw

    # region book:promotion-causal-naive-impact
    def naive_promotion_impact(self) -> NaiveImpact:
        """Calculate naive promotion impact (ignoring confounders)"""
        # Group by promotion status and calculate mean sales
        impact = self.analysis_data.groupby("on_promotion")["sales_units"].mean().reset_index()
        # Calculate lift
        no_promo = impact.loc[~impact["on_promotion"], "sales_units"].values[0]
        promo = impact.loc[impact["on_promotion"], "sales_units"].values[0]
        lift = promo - no_promo
        percent_lift = (promo / no_promo - 1) * 100
        return NaiveImpact(
            no_promo_sales=float(no_promo),
            promo_sales=float(promo),
            absolute_lift=float(lift),
            percent_lift=float(percent_lift),
        )

    # endregion book:promotion-causal-naive-impact

    # region book:promotion-causal-regression
    def regression_adjustment(self) -> RegressionAdjustment:
        """Estimate promotion impact using regression adjustment for confounders"""
        if self.analysis_data["on_promotion"].nunique() != 2:
            raise ValueError("Effect estimation requires both promotion groups.")
        # Prepare features
        X = self.analysis_data[
            [
                "on_promotion",
                "day_of_week",
                "month",
                "weekend",
                "holiday",
                "product_category",
                "store_tier",
                "store_traffic",
            ]
        ]
        # Convert categorical variables to dummies
        X = pd.get_dummies(X, columns=["day_of_week", "month", "product_category", "store_tier"])
        # Add intercept
        X = sm.add_constant(X.astype(float), has_constant="add")
        # Target variable
        y = self.analysis_data["sales_units"]
        # Fit model
        model = sm.OLS(y, X).fit()
        # Extract promotion effect
        promotion_effect = model.params["on_promotion"]
        promotion_pvalue = model.pvalues["on_promotion"]
        return RegressionAdjustment(
            promotion_effect=float(promotion_effect),
            p_value=float(promotion_pvalue),
            r_squared=float(model.rsquared),
        )

    # endregion book:promotion-causal-regression

    # region book:promotion-causal-matching
    def matching_analysis(self, max_distance: float = 0.1) -> MatchingImpact:
        """Estimate promotion impact using propensity score matching"""
        from sklearn.linear_model import LogisticRegression

        # Features for propensity model
        # Discounted price is a mediator; store_traffic must be measured before treatment.
        X = self.analysis_data[
            [
                "day_of_week",
                "month",
                "weekend",
                "holiday",
                "product_category",
                "store_tier",
                "store_traffic",
            ]
        ]
        # Convert categorical variables to dummies
        X = pd.get_dummies(X, columns=["day_of_week", "month", "product_category", "store_tier"])
        # Treatment variable
        y = self.analysis_data["on_promotion"]
        # Fit propensity model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        # Calculate propensity scores
        prop_scores = model.predict_proba(X)[:, 1]
        # Add scores to data
        self.analysis_data["propensity_score"] = prop_scores
        # Perform matching
        treated = self.analysis_data[self.analysis_data["on_promotion"]].copy()
        control = self.analysis_data[~self.analysis_data["on_promotion"]].copy()
        # Find matches
        matched_pairs = []
        for _, treat_row in treated.iterrows():
            # Find control units within max_distance
            control_matches = control[
                abs(control["propensity_score"] - treat_row["propensity_score"]) <= max_distance
            ]
            if len(control_matches) > 0:
                # Pick closest match
                best_match = control_matches.iloc[
                    (control_matches["propensity_score"] - treat_row["propensity_score"]).abs().argmin()
                ]
                matched_pairs.append((treat_row, best_match))
        # endregion book:promotion-causal-matching

        # region book:promotion-causal-matching-effect
        # Calculate treatment effect from matched pairs
        if matched_pairs:
            treatment_outcomes = np.array([pair[0]["sales_units"] for pair in matched_pairs])
            control_outcomes = np.array([pair[1]["sales_units"] for pair in matched_pairs])
            effect = np.mean(treatment_outcomes - control_outcomes)
            percent_effect = np.mean((treatment_outcomes / control_outcomes - 1) * 100)

            return MatchingImpact(
                matched_pairs=len(matched_pairs),
                average_treatment_effect=float(effect),
                percent_effect=float(percent_effect),
            )
        raise ValueError("No eligible matches; effect is not estimable.")
        # endregion book:promotion-causal-matching-effect

    # region book:promotion-causal-double-ml
    def double_ml_forest(self) -> DoubleMLResult:
        """Estimate heterogeneous treatment effects using double ML causal forest"""
        from econml.dml import CausalForestDML

        # Prepare data
        df = self.analysis_data.copy()
        # Treatment variable
        T = df["on_promotion"].astype(float).values
        # Outcome variable
        Y = df["sales_units"].values
        # Features for effect estimation
        X = df[
            [
                "day_of_week",
                "month",
                "weekend",
                "holiday",
                "store_traffic",
            ]
        ]
        # Encode categorical variables
        W = pd.get_dummies(df[["product_category", "store_tier"]])

        # Fit causal forest model
        cf = CausalForestDML(
            n_estimators=100,
            min_samples_leaf=10,
            max_depth=5,
            random_state=42,
            discrete_treatment=True,
        )
        cf.fit(Y, T, X=X, W=W)
        # endregion book:promotion-causal-double-ml

        # region book:promotion-causal-double-ml-ate
        # Get overall average treatment effect
        ate = cf.ate(X=X)
        # Generate heterogeneous treatment effects
        cate_estimates = cf.effect(X=X)
        # Analyze heterogeneity by product category and store tier
        df["cate"] = cate_estimates
        # Keep the original labels; dummy columns exist only in W.
        df["original_category"] = df["product_category"]
        df["original_tier"] = df["store_tier"]
        # endregion book:promotion-causal-double-ml-ate

        # region book:promotion-causal-double-ml-by-segment
        # Calculate treatment effects by category
        category_effects = df.groupby("original_category")["cate"].mean().to_dict()
        # Calculate treatment effects by store tier
        tier_effects = df.groupby("original_tier")["cate"].mean().to_dict()
        return DoubleMLResult(
            average_treatment_effect=float(ate),
            heterogeneous_effects=SegmentEffects(
                by_category=category_effects,
                by_store_tier=tier_effects,
            ),
        )
        # endregion book:promotion-causal-double-ml-by-segment

    # region book:promotion-causal-dowhy
    def dowhy_analysis(self) -> DoWhyResult:
        """Estimate causal effect using the DoWhy causal inference framework"""
        from dowhy import CausalModel

        # Identify variables from our causal graph
        treatment = "on_promotion"
        outcome = "sales_units"
        # Convert our internal graph to DoWhy format
        graph = self.causal_graph.subgraph(self.analysis_data.columns).copy()
        # Create DoWhy model
        model = CausalModel(
            data=self.analysis_data,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )
        # Identify effect
        identified_estimand = model.identify_effect()
        # Estimate effect using regression adjustment
        estimate_regression = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression",
        )
        # Estimate effect using matching
        estimate_matching = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.propensity_score_matching",
        )
        # endregion book:promotion-causal-dowhy

        # region book:promotion-causal-dowhy-refute
        # Perform refutation tests
        refute_random = model.refute_estimate(
            identified_estimand,
            estimate_regression,
            method_name="random_common_cause",
        )
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_regression,
            method_name="placebo_treatment_refuter",
        )

        return DoWhyResult(
            regression_effect=float(estimate_regression.value),
            matching_effect=float(estimate_matching.value),
            random_refutation=str(refute_random),
            placebo_refutation=str(refute_placebo),
        )
        # endregion book:promotion-causal-dowhy-refute

    # region book:promotion-causal-counterfactual
    def perform_counterfactual_analysis(self, scenario: CounterfactualScenario) -> CounterfactualResult:
        """Predict outcomes under counterfactual scenarios"""
        if self.analysis_data["on_promotion"].nunique() != 2:
            raise ValueError("Effect estimation requires both promotion groups.")
        # Create a copy of the analysis data
        cf_data = self.analysis_data.copy()
        # Apply counterfactual scenario changes
        for key, value in scenario.overrides.items():
            if key in cf_data.columns:
                cf_data[key] = value
        # Use the same pre-treatment covariates as regression adjustment.
        features = [
            "on_promotion",
            "day_of_week",
            "month",
            "weekend",
            "holiday",
            "store_traffic",
            "product_category",
            "store_tier",
        ]
        unsupported = set(scenario.overrides) - set(features)
        if unsupported:
            return CounterfactualResult(error=f"Counterfactual features not modeled: {sorted(unsupported)}")
        categorical = ["day_of_week", "month", "product_category", "store_tier"]
        actual_features = pd.get_dummies(self.analysis_data[features], columns=categorical, dtype=float)
        X = pd.get_dummies(cf_data[features], columns=categorical, dtype=float)
        X = X.reindex(columns=actual_features.columns, fill_value=0)
        # Add intercept
        X = sm.add_constant(X.astype(float), has_constant="add")
        # Train regression model on original data
        y = self.analysis_data["sales_units"]
        model = sm.OLS(y, sm.add_constant(actual_features.astype(float), has_constant="add")).fit()
        # endregion book:promotion-causal-counterfactual

        # region book:promotion-causal-counterfactual-predict
        # Predict counterfactual outcomes
        try:
            cf_predictions = model.predict(X)
            # Calculate summary statistics
            cf_results = CounterfactualSummary(
                mean_predicted_sales=float(cf_predictions.mean()),
                total_predicted_sales=float(cf_predictions.sum()),
                min_predicted_sales=float(cf_predictions.min()),
                max_predicted_sales=float(cf_predictions.max()),
                actual_total_sales=0.0,
                percentage_change=0.0,
            )
            # Compare with actual
            actual_total = self.analysis_data["sales_units"].sum()
            percentage_change = (cf_results.total_predicted_sales - actual_total) / actual_total * 100
            cf_results = cf_results.model_copy(
                update={
                    "actual_total_sales": float(actual_total),
                    "percentage_change": float(percentage_change),
                }
            )
            return CounterfactualResult(summary=cf_results)
        except Exception as e:
            return CounterfactualResult(error=str(e))
        # endregion book:promotion-causal-counterfactual-predict

    # region book:promotion-causal-roi
    def calculate_promotion_roi(self, promotion_cost: float) -> PromotionROI:
        """Calculate ROI of promotions considering causal effects"""
        # Get causal effect estimate
        causal_effect = self.regression_adjustment()
        # Get product price and margin data
        promoted = self.analysis_data["on_promotion"]
        avg_price = self.analysis_data.loc[promoted, "price"].mean()
        avg_margin_percent = 0.35  # Placeholder - would come from actual data
        # Calculate incremental units
        incremental_units = causal_effect.promotion_effect * promoted.sum()
        # Calculate incremental revenue
        incremental_revenue = incremental_units * avg_price
        # Calculate incremental profit
        incremental_profit = incremental_revenue * avg_margin_percent
        # Calculate ROI
        roi = (incremental_profit - promotion_cost) / promotion_cost
        return PromotionROI(
            incremental_revenue=float(incremental_revenue),
            incremental_profit=float(incremental_profit),
            roi_percent=float(roi * 100),
        )

    # endregion book:promotion-causal-roi


# region book:promotion-causal-example-setup
# Example usage
if __name__ == "__main__":
    # This would be replaced with actual data in a real implementation
    # Simulating some sample data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", end="2023-03-31")
    stores = range(1, 11)
    products = range(1, 21)
    # Generate sample data
    data = []
    # endregion book:promotion-causal-example-setup

    # region book:promotion-causal-example-loop
    for date in dates:
        for store in stores:
            for product in products:
                # Determine if product is on promotion (20% chance)
                on_promotion = np.random.random() < 0.2
                # Base demand
                base_demand = 10 + product * 0.5
                # Store traffic effect
                store_effect = 1 + store / 20
                # Product effect
                product_effect = 1 + product / 30
                # Day of week effect
                dow_effect = 1 + 0.1 * (date.dayofweek >= 5)  # Higher on weekends
                # Promotion effect
                promo_effect = 1.3 if on_promotion else 1.0
                # Price (affected by promotion)
                regular_price = 9.99 + product * 0.5
                price = regular_price * (1 - 0.2 * on_promotion)
                # Store traffic
                store_traffic = np.random.poisson(100) * (1 + 0.1 * (date.dayofweek >= 5))
                # Final sales
                sales = base_demand * store_effect * product_effect * dow_effect * promo_effect
                sales = np.random.poisson(sales)
                # Product category
                product_category = f"Category {(product - 1) // 5 + 1}"
                # Store tier
                store_tier = f"Tier {(store - 1) // 3 + 1}"
                # Add record
                data.append(
                    {
                        "date": date,
                        "product_id": product,
                        "store_id": store,
                        "sales_units": sales,
                        "on_promotion": on_promotion,
                        "price": price,
                        "product_category": product_category,
                        "store_tier": store_tier,
                        "store_traffic": store_traffic,
                    }
                )
    # endregion book:promotion-causal-example-loop

    # region book:promotion-causal-example-dataframes
    sales_df = pd.DataFrame(data)
    # Create other necessary DataFrames
    product_df = pd.DataFrame(
        {
            "product_id": range(1, 21),
            "product_category": [f"Category {(p - 1) // 5 + 1}" for p in range(1, 21)],
        }
    )
    store_df = pd.DataFrame(
        {
            "store_id": range(1, 11),
            "store_tier": [f"Tier {(s - 1) // 3 + 1}" for s in range(1, 11)],
        }
    )
    promotion_df = sales_df[["date", "product_id", "store_id", "on_promotion"]].copy()

    # Initialize analyzer
    analyzer = PromotionCausalAnalyzer(sales_df, product_df, store_df, promotion_df)
    # endregion book:promotion-causal-example-dataframes

    # region book:promotion-causal-example-output
    # Visualize causal graph
    analyzer.visualize_causal_graph("promotion_causal_graph.png")
    # Calculate ROI
    roi_result = analyzer.calculate_promotion_roi(promotion_cost=1000)
    print(f"Promotion ROI: {roi_result.roi_percent:.2f}%")
    # Counterfactual scenario: What if we ran promotions only on weekends?
    counterfactual = analyzer.perform_counterfactual_analysis(
        CounterfactualScenario(overrides={"on_promotion": analyzer.analysis_data["weekend"] == 1})
    )
    if counterfactual.summary:
        print(
            f"Counterfactual Analysis: {counterfactual.summary.percentage_change:.2f}% change in total sales"
        )
    else:
        print(f"Counterfactual Analysis Error: {counterfactual.error}")
    # endregion book:promotion-causal-example-output
