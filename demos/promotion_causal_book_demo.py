"""The ROI proxy excludes baseline discount losses, funding, and cannibalization."""

from math import isfinite

# region book:promotion-causal-imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import BaseModel, ConfigDict
from statsmodels.regression.linear_model import RegressionResultsWrapper

# endregion book:promotion-causal-imports


# region book:promotion-causal-schemas
class NaiveImpact(BaseModel):
    no_promo_sales: float
    promo_sales: float
    absolute_lift: float
    percent_lift: float | None


class RegressionAdjustment(BaseModel):
    promotion_effect: float
    p_value: float
    r_squared: float
    confidence_interval: tuple[float, float]


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


class MatchingImpact(BaseModel):
    matched_pairs: int
    matched_treated_effect: float
    percent_effect: float | None
    confidence_interval: tuple[float, float] | None
    matched_treated_fraction: float
    propensity_overlap: tuple[float, float]
    balance_before: dict[str, float]
    balance_after: dict[str, float]


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
        self._validate_measurements(df)
        df["on_promotion"] = df["on_promotion"].astype(bool)
        # Create calendar features
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["holiday"] = self._is_holiday(df["date"])
        # endregion book:promotion-causal-prepare-data-promo

        # region book:promotion-causal-prepare-data-lags
        # Calendar lags preserve missing days and initial unavailability as NaN.
        df = df.sort_values(["store_id", "product_id", "date"], kind="stable").reset_index(drop=True)
        indexed = df.set_index(["product_id", "store_id", "date"])
        if not indexed.index.is_unique:
            raise ValueError("Analysis requires one observation per product-store-day")
        for lag in [1, 2, 3, 7, 14]:
            keys = pd.MultiIndex.from_arrays([df.product_id, df.store_id, df.date - pd.Timedelta(days=lag)])
            for column in ("sales_units", "on_promotion"):
                prefix = "sales" if column == "sales_units" else column
                df[f"{prefix}_lag_{lag}"] = indexed[column].astype(float).reindex(keys).to_numpy()
        return df
        # endregion book:promotion-causal-prepare-data-lags

    @staticmethod
    def _validate_measurements(df: pd.DataFrame) -> None:
        required = [
            "date",
            "product_id",
            "store_id",
            "sales_units",
            "price",
            "store_traffic",
            "product_category",
            "store_tier",
            "on_promotion",
        ]
        missing = df[required].isna().any()
        if missing.any():
            raise ValueError(f"Analysis data contain missing values in {list(missing[missing].index)}")
        numeric = df[["sales_units", "price", "store_traffic"]].to_numpy(dtype=float)
        if not np.isfinite(numeric).all() or (numeric < 0).any():
            raise ValueError("Analysis measurements must be finite and nonnegative")

    # region book:promotion-causal-is-holiday
    def _is_holiday(self, dates: pd.Series) -> pd.Series:
        """Determine if dates are holidays"""
        # This is a simplified placeholder - in a real system,
        # you would use a holiday calendar library or a lookup table
        holidays = ["2023-01-01", "2023-12-25"]  # Example holidays
        return dates.isin(pd.to_datetime(holidays))

    # endregion book:promotion-causal-is-holiday

    # region book:promotion-causal-define-graph-joined
    def _define_causal_graph(self) -> nx.DiGraph:
        """Define the directed acyclic graph of causal relationships"""
        G = nx.DiGraph()
        # Add nodes
        nodes = [
            "on_promotion",
            "sales_units",
            "price",
            "day_of_week",
            "weekend",
            "store_traffic",
            "product_category",
            "store_tier",
        ]
        G.add_nodes_from(nodes)

        # Add edges (causal relationships)
        edges = [
            ("day_of_week", "weekend"),
            ("weekend", "store_traffic"),
            ("product_category", "price"),
            ("on_promotion", "price"),
            ("price", "sales_units"),
            ("on_promotion", "sales_units"),
        ]
        for covariate in ("weekend", "store_traffic", "product_category", "store_tier"):
            edges.extend([(covariate, "on_promotion"), (covariate, "sales_units")])
        G.add_edges_from(edges)

        return G
        # endregion book:promotion-causal-define-graph-joined

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
        percent_lift = (promo / no_promo - 1) * 100 if no_promo > 0 else None
        return NaiveImpact(
            no_promo_sales=float(no_promo),
            promo_sales=float(promo),
            absolute_lift=float(lift),
            percent_lift=percent_lift,
        )

    # endregion book:promotion-causal-naive-impact

    # region book:promotion-causal-regression
    def regression_adjustment(self) -> RegressionAdjustment:
        """Estimate promotion impact using regression adjustment for confounders"""
        if self.analysis_data["on_promotion"].nunique() != 2:
            raise ValueError("Effect estimation requires both promotion groups.")
        model = self._fit_promotion_model()
        treated = self.analysis_data.assign(on_promotion=True)
        control = self.analysis_data.assign(on_promotion=False)
        contrast = (self._promotion_features(treated) - self._promotion_features(control)).mean().to_numpy()
        estimate = model.t_test(contrast)
        return RegressionAdjustment(
            promotion_effect=float(np.asarray(estimate.effect).item()),
            p_value=float(np.asarray(estimate.pvalue).item()),
            r_squared=float(model.rsquared),
            confidence_interval=tuple(float(value) for value in estimate.conf_int()[0]),
        )

    # endregion book:promotion-causal-regression

    def _fit_promotion_model(self) -> RegressionResultsWrapper:
        if self.analysis_data["on_promotion"].nunique() != 2:
            raise ValueError("Effect estimation requires both promotion groups.")
        return sm.OLS(self.analysis_data["sales_units"], self._promotion_features(self.analysis_data)).fit(
            cov_type="HC3"
        )

    def _promotion_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = [
            "on_promotion",
            "day_of_week",
            "month",
            "holiday",
            "product_category",
            "store_tier",
            "store_traffic",
        ]
        X = pd.get_dummies(
            data[features],
            columns=["day_of_week", "month", "product_category", "store_tier"],
            drop_first=True,
        ).astype(float)
        for column in list(X):
            if column.startswith("product_category_"):
                X[f"promotion_x_{column}"] = X[column] * X["on_promotion"]
        return sm.add_constant(X, has_constant="add")

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
                control = control.drop(index=best_match.name)  # No reuse of outcome units.
        # endregion book:promotion-causal-matching

        # region book:promotion-causal-matching-effect
        # Calculate treatment effect from matched pairs
        if matched_pairs:
            treatment_outcomes = np.array([pair[0]["sales_units"] for pair in matched_pairs])
            control_outcomes = np.array([pair[1]["sales_units"] for pair in matched_pairs])
            effect = np.mean(treatment_outcomes - control_outcomes)
            return MatchingImpact(
                matched_pairs=len(matched_pairs),
                matched_treated_effect=float(effect),
                **self._matching_diagnostics(matched_pairs, X, y, prop_scores, effect),
            )
        raise ValueError("No eligible matches; effect is not estimable.")
        # endregion book:promotion-causal-matching-effect

    @staticmethod
    def _matching_diagnostics(matched_pairs, X, y, prop_scores, effect):
        treatment_outcomes = np.array([pair[0]["sales_units"] for pair in matched_pairs])
        control_outcomes = np.array([pair[1]["sales_units"] for pair in matched_pairs])
        percent_effect = (
            float(np.mean((treatment_outcomes / control_outcomes - 1) * 100))
            if np.all(control_outcomes > 0)
            else None
        )
        differences = treatment_outcomes - control_outcomes
        # Conditional paired interval for independent synthetic units; refit/bootstrap for deployment.
        se = float(np.std(differences, ddof=1) / np.sqrt(len(differences))) if len(differences) > 1 else None
        treated_X, control_X = X.loc[y], X.loc[~y]
        pooled = X.std(ddof=0).replace(0, 1)
        before = ((treated_X.mean() - control_X.mean()) / pooled).to_dict()
        matched_t = X.loc[[pair[0].name for pair in matched_pairs]]
        matched_c = X.loc[[pair[1].name for pair in matched_pairs]]
        after = ((matched_t.mean() - matched_c.mean()) / pooled).to_dict()
        overlap = (
            max(prop_scores[y].min(), prop_scores[~y].min()),
            min(prop_scores[y].max(), prop_scores[~y].max()),
        )
        return {
            "percent_effect": percent_effect,
            "confidence_interval": (float(effect) - 1.96 * se, float(effect) + 1.96 * se)
            if se is not None
            else None,
            "matched_treated_fraction": len(matched_pairs) / int(y.sum()),
            "propensity_overlap": overlap,
            "balance_before": before,
            "balance_after": after,
        }

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
        X = pd.get_dummies(
            df[
                [
                    "day_of_week",
                    "month",
                    "weekend",
                    "holiday",
                    "store_traffic",
                    "product_category",
                    "store_tier",
                ]
            ],
            columns=["product_category", "store_tier"],
            drop_first=True,
        ).astype(float)
        # Fit causal forest model
        cf = CausalForestDML(
            n_estimators=100,
            min_samples_leaf=10,
            max_depth=5,
            random_state=42,
            discrete_treatment=True,
        )
        cf.fit(Y, T, X=X)
        # endregion book:promotion-causal-double-ml

        # region book:promotion-causal-double-ml-ate
        # Get overall average treatment effect
        ate = cf.ate(X=X)
        # Generate heterogeneous treatment effects
        cate_estimates = cf.effect(X=X)
        # Analyze heterogeneity by product category and store tier
        df["cate"] = cate_estimates
        # Keep original labels for segment summaries; X contains the dummy columns.
        df["original_category"] = df["product_category"]
        df["original_tier"] = df["store_tier"]
        # endregion book:promotion-causal-double-ml-ate

        # region book:promotion-causal-double-ml-by-segment
        # Segment averages include modeled category and tier effect modifiers.
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
            random_state=42,
        )
        refute_placebo = model.refute_estimate(
            identified_estimand,
            estimate_regression,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",  # This binary-treatment path uses DoWhy's local seeded RNG.
            random_state=42,
        )

        return DoWhyResult(
            regression_effect=float(estimate_regression.value),
            matching_effect=float(estimate_matching.value),
            random_refutation=str(refute_random),
            placebo_refutation=str(refute_placebo),
        )
        # endregion book:promotion-causal-dowhy-refute

    # region book:promotion-causal-counterfactual-joined
    def perform_counterfactual_analysis(self, scenario: CounterfactualScenario) -> CounterfactualResult:
        """Predict outcomes under counterfactual scenarios"""
        if self.analysis_data["on_promotion"].nunique() != 2:
            raise ValueError("Effect estimation requires both promotion groups.")
        # Create a copy of the analysis data
        cf_data = self.analysis_data.copy()
        if set(scenario.overrides) != {"on_promotion"}:
            return CounterfactualResult(error="Only the on_promotion policy intervention is supported")
        values = scenario.overrides["on_promotion"]
        if isinstance(values, pd.Series) and not values.index.equals(cf_data.index):
            return CounterfactualResult(error="Promotion policy must align with analysis rows")
        if not np.isin(np.asarray(values), [0, 1]).all():
            return CounterfactualResult(error="Promotion policy must be binary and nonmissing")
        # Apply counterfactual scenario changes
        for key, value in scenario.overrides.items():
            if key in cf_data.columns:
                cf_data[key] = value
        X = self._promotion_features(cf_data)
        model = self._fit_promotion_model()

        try:
            cf_predictions = model.predict(X)
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
            if actual_total <= 0:
                error = "Actual sales total must be positive for a relative comparison"
                return CounterfactualResult(error=error)
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
        # endregion book:promotion-causal-counterfactual-joined

    # region book:promotion-causal-roi
    def calculate_promotion_roi(self, promotion_cost: float) -> PromotionROI:
        """Illustrative proxy weighting promoted row effects and prices, with a fixed 35% margin."""
        if not isfinite(promotion_cost) or promotion_cost <= 0:
            raise ValueError("Promotion cost must be finite and positive")
        model = self._fit_promotion_model()
        promoted = self.analysis_data["on_promotion"]
        treated = self._promotion_features(self.analysis_data.assign(on_promotion=True))
        control = self._promotion_features(self.analysis_data.assign(on_promotion=False))
        incremental_units = (treated - control) @ model.params
        # Preserve effect/price covariance by monetizing each promoted row, not population means.
        incremental_revenue = (incremental_units[promoted] * self.analysis_data.loc[promoted, "price"]).sum()
        avg_margin_percent = 0.35  # Illustrative proportional unit margin, not an accounting contract.
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


def simulate_promotion_data(seed: int = 42) -> pd.DataFrame:
    """Independent synthetic store-product-days; population ATE is 9.75 units.

    Traffic is pretreatment. No carryover, censoring, or interference is simulated.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for date in pd.date_range("2023-01-01", "2023-03-31"):
        weekend = int(date.dayofweek >= 5)
        for store in range(1, 11):
            tier = (store - 1) // 3 + 1
            for product in range(1, 21):
                category = (product - 1) // 5 + 1
                traffic = rng.poisson(100 + 20 * weekend)
                propensity = 1 / (
                    1 + np.exp(-(-2 + 0.015 * (traffic - 100) + 0.3 * category + 0.15 * tier + 0.4 * weekend))
                )
                promotion = rng.random() < propensity
                regular_price = 10 + 2 * category
                price = regular_price * (1 - 0.2 * promotion)
                untreated = 10 + 0.12 * traffic + 2 * category + 3 * tier + 2 * weekend
                effect = 4 + 0.5 * category + 1.5 * 0.2 * regular_price
                sales_mean = untreated + (4 + 0.5 * category) * promotion - 1.5 * (price - regular_price)
                rows.append(
                    dict(
                        date=date,
                        product_id=product,
                        store_id=store,
                        sales_units=rng.poisson(sales_mean),
                        on_promotion=promotion,
                        price=price,
                        product_category=f"Category {category}",
                        store_tier=f"Tier {tier}",
                        store_traffic=traffic,
                        untreated_mean=untreated,
                        treatment_effect=effect,
                    )
                )
    return pd.DataFrame(rows)


def _print_effect_estimates(analyzer, sales_df):
    truth = sales_df["treatment_effect"].mean()
    treated_truth = sales_df.loc[sales_df.on_promotion, "treatment_effect"].mean()
    adjusted = analyzer.regression_adjustment()
    print(
        f"Known ATE={truth:.3f}; ATT={treated_truth:.3f}; adjusted={adjusted.promotion_effect:.3f}; "
        f"95% HC3 interval={adjusted.confidence_interval}"
    )
    matching = analyzer.matching_analysis()
    print(f"Matching percent effect={matching.percent_effect:.3f}")
    forest = analyzer.double_ml_forest()
    print(f"Forest ATE={forest.average_treatment_effect:.3f}")
    dowhy = analyzer.dowhy_analysis()
    print(f"DoWhy regression={dowhy.regression_effect:.3f}; matching={dowhy.matching_effect:.3f}")
    print(dowhy.random_refutation)
    print(dowhy.placebo_refutation)


# region book:promotion-causal-example-setup-joined
# Example usage
if __name__ == "__main__":
    # The seeded simulator below supplies the complete synthetic fixture.

    data = simulate_promotion_data(seed=42).to_dict("records")

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

    # Visualize causal graph
    analyzer.visualize_causal_graph("promotion_causal_graph.png")
    naive = analyzer.naive_promotion_impact()
    print(f"Naive lift={naive.absolute_lift:.3f}")
    _print_effect_estimates(analyzer, sales_df)
    # Calculate ROI
    roi_result = analyzer.calculate_promotion_roi(promotion_cost=1000)
    print(f"Promotion ROI: {roi_result.roi_percent:.2f}% (illustrative proxy, not approval)")
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
    # endregion book:promotion-causal-example-setup-joined
