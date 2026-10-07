import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from agents.promotion_causal import PromotionCausalAnalyzer

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["library", "book"])
def implementation(request):
    if request.param == "library":
        return PromotionCausalAnalyzer, "promotion_applied", "sales"
    module = runpy.run_path(str(ROOT / "demos/promotion_causal_book_demo.py"))
    return module["PromotionCausalAnalyzer"], "on_promotion", "sales_units"


def analyze(implementation, sales, promotions=None, products=None, stores=None):
    analyzer_type, treatment, outcome = implementation
    sales = sales.rename(columns={"on_promotion": treatment, "sales_units": outcome})
    if promotions is None:
        promotions = sales[["date", "product_id", "store_id", treatment]].copy()
    else:
        promotions = promotions.rename(columns={"on_promotion": treatment})
    if products is None:
        products = sales[["product_id", "product_category"]].drop_duplicates()
    if stores is None:
        stores = sales[["store_id", "store_tier"]].drop_duplicates()
    return analyzer_type(sales, products, stores, promotions)


@pytest.fixture
def sales():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-02", "2023-01-01"]),
            "product_id": ["P1", "P1"],
            "store_id": ["S1", "S1"],
            "sales_units": [18.0, 10.0],
            "on_promotion": [True, False],
            "product_category": ["Cat1", "Cat1"],
            "store_tier": ["Tier1", "Tier1"],
            "store_traffic": [100, 100],
            "price": [8.0, 10.0],
        }
    )


def test_treatment_and_metadata_survive_merges(implementation, sales, caplog):
    original = sales.copy(deep=True)
    with caplog.at_level("INFO", logger="agents.promotion_causal"):
        analyzer = analyze(implementation, sales)
    _, treatment, _ = implementation
    result = analyzer.analysis_data.sort_values("date")
    assert result[treatment].tolist() == [0, 1]
    assert result["product_category"].tolist() == ["Cat1", "Cat1"]
    assert result["store_tier"].tolist() == ["Tier1", "Tier1"]
    assert not any(c.endswith(("_x", "_y")) for c in result.columns)
    if treatment == "promotion_applied":
        assert "Columns:" in caplog.text
    pd.testing.assert_frame_equal(sales, original)


@pytest.mark.parametrize("table", ["promotions", "products", "stores"])
def test_duplicate_join_keys_are_rejected(implementation, sales, table):
    tables = {
        "promotions": sales[["date", "product_id", "store_id", "on_promotion"]],
        "products": sales[["product_id", "product_category"]].drop_duplicates(),
        "stores": sales[["store_id", "store_tier"]].drop_duplicates(),
    }
    duplicate = pd.concat([tables[table], tables[table].iloc[:1]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        analyze(implementation, sales, **{table: duplicate})


def test_disagreeing_treatment_is_rejected(implementation, sales):
    promotions = sales[["date", "product_id", "store_id", "on_promotion"]].copy()
    promotions["on_promotion"] = ~promotions["on_promotion"]
    with pytest.raises(ValueError, match="treatment|promotion"):
        analyze(implementation, sales, promotions=promotions)


def test_promotion_dates_are_normalized_without_mutating_input(implementation, sales):
    promotions = sales[["date", "product_id", "store_id", "on_promotion"]].copy()
    promotions["date"] = promotions["date"].dt.strftime("%Y-%m-%d")
    original = promotions.copy(deep=True)
    analyzer = analyze(implementation, sales.drop(columns="on_promotion"), promotions=promotions)
    _, treatment, _ = implementation
    assert analyzer.analysis_data[treatment].sum() == 1
    pd.testing.assert_frame_equal(promotions, original)


@pytest.mark.parametrize("implementation", ["book"], indirect=True)
def test_lags_use_past_rows_of_the_same_store_and_product(implementation, sales):
    records = []
    for store, product, offset in [("S1", "P1", 0), ("S2", "P1", 100), ("S1", "P2", 200)]:
        for day in range(1, 4):
            row = sales.iloc[0].to_dict()
            row.update(date=f"2023-01-0{day}", store_id=store, product_id=product, sales_units=offset + day)
            records.append(row)
    shuffled = pd.DataFrame(records).sample(frac=1, random_state=42)
    data = analyze(implementation, shuffled).analysis_data
    for store, product, expected in [
        ("S1", "P1", [0, 1, 2]),
        ("S2", "P1", [0, 101, 102]),
        ("S1", "P2", [0, 201, 202]),
    ]:
        group = data[(data["store_id"] == store) & (data["product_id"] == product)].sort_values("date")
        assert group["sales_lag_1"].tolist() == expected


@pytest.fixture
def known_effect_sales():
    # Price is a post-treatment mediator; adjusting for it hides the total effect.
    rng = np.random.default_rng(20261002)
    size = 1200
    dates = pd.date_range("2023-01-01", periods=size)
    category = rng.integers(0, 2, size)
    traffic = rng.normal(100, 10, size)
    treatment = rng.binomial(1, 0.15 + 0.25 * category + 0.15 * (dates.dayofweek >= 5))
    return pd.DataFrame(
        {
            "date": dates,
            "product_id": category,
            "store_id": "S1",
            "sales_units": 20 + 4 * category + 0.1 * traffic + 8 * treatment,
            "on_promotion": treatment.astype(bool),
            "product_category": np.where(category, "Cat2", "Cat1"),
            "store_tier": "Tier1",
            "store_traffic": traffic,
            "price": 10 - 2 * treatment,
        }
    )


def test_regression_recovers_total_effect(implementation, known_effect_sales):
    analyzer = analyze(implementation, known_effect_sales)
    result = analyzer.regression_adjustment()
    effect = result["estimated_ATE"] if isinstance(result, dict) else result.promotion_effect
    assert effect == pytest.approx(8.0, abs=0.01)
    if hasattr(analyzer, "common_causes"):
        assert "price" not in analyzer.common_causes
        assert "product_category" in analyzer.common_causes
        assert implementation[1] not in result["confounders_used"]


@pytest.mark.parametrize("method", ["regression_adjustment", "matching_analysis"])
def test_estimator_recovers_generator_known_effect(implementation, method):
    # Load the owned module directly until the environment job fixes utils/__init__.py.
    generate = runpy.run_path(str(ROOT / "utils/data_generation.py"))["generate_synthetic_retail_data"]
    data, _, _ = generate(
        end_date_str="2024-12-31",
        num_stores=1,
        num_products=1,
        seed=42,
        seasonal_effect_amplitude=0,
        weekend_sales_effect=0,
        promo_base_prob=0.4,
        promo_weekend_add_prob=0,
        promo_cat2_add_prob=0,
        promo_store1_add_prob=0,
        noise_std_dev=0,
        traffic_sales_effect_divisor=float("inf"),
        true_promo_effect_multiplier=1.5,
    )
    result = getattr(analyze(implementation, data), method)()
    if isinstance(result, dict):
        assert "error" not in result, result
        effect = result["estimated_ATE"]
    elif method == "regression_adjustment":
        effect = result.promotion_effect
    else:
        effect = result.average_treatment_effect
    # E[Y(0)] = Poisson mean 12 * Tier1 multiplier 1.1; lift = 0.5 * 13.2.
    # Seed 42, 731 observations: tolerate count noise, not a halved/zero effect.
    assert effect == pytest.approx(6.6, abs=1.3)


def test_counterfactual_recovers_all_treated_mean(implementation, known_effect_sales):
    analyzer = analyze(implementation, known_effect_sales)
    _, treatment, _ = implementation
    scenario = {treatment: 1}
    if treatment == "on_promotion":
        scenario_type = analyzer.perform_counterfactual_analysis.__globals__["CounterfactualScenario"]
        scenario = scenario_type(overrides=scenario)
    result = analyzer.perform_counterfactual_analysis(scenario)
    expected = (known_effect_sales["sales_units"] + 8 * (~known_effect_sales["on_promotion"])).mean()
    if isinstance(result, dict):
        assert "error" not in result, result
        actual = result["counterfactual_mean_sales"]
    else:
        assert result.error is None
        actual = result.summary.mean_predicted_sales
    assert actual == pytest.approx(expected, abs=0.01)


def test_counterfactual_rejects_unmodeled_price_intervention(implementation, known_effect_sales):
    analyzer = analyze(implementation, known_effect_sales)
    scenario = {"price": 20.0}
    if implementation[1] == "on_promotion":
        schema = analyzer.perform_counterfactual_analysis.__globals__["CounterfactualScenario"]
        scenario = schema(overrides=scenario)
    result = analyzer.perform_counterfactual_analysis(scenario)
    assert result.get("error") if isinstance(result, dict) else result.error


@pytest.mark.parametrize("treated", [False, True])
@pytest.mark.parametrize("method", ["regression_adjustment", "perform_counterfactual_analysis"])
def test_effect_estimation_requires_both_treatment_groups(
    implementation, known_effect_sales, treated, method
):
    known_effect_sales["on_promotion"] = treated
    analyzer = analyze(implementation, known_effect_sales)
    args = []
    if method == "perform_counterfactual_analysis":
        scenario = {implementation[1]: int(not treated)}
        if implementation[1] == "on_promotion":
            schema = analyzer.perform_counterfactual_analysis.__globals__["CounterfactualScenario"]
            scenario = schema(overrides=scenario)
        args = [scenario]
    if implementation[1] == "on_promotion":
        with pytest.raises(ValueError, match="promotion.*groups"):
            getattr(analyzer, method)(*args)
    else:
        result = getattr(analyzer, method)(*args)
        assert "error" in result


def test_book_import_does_not_require_optional_estimators(monkeypatch):
    monkeypatch.setitem(sys.modules, "dowhy", None)
    monkeypatch.setitem(sys.modules, "econml.dml", None)
    runpy.run_path(str(ROOT / "demos/promotion_causal_book_demo.py"))


@pytest.mark.parametrize("implementation", ["book"], indirect=True)
def test_book_roi_scales_incremental_units_to_promoted_rows(implementation, known_effect_sales):
    analyzer = analyze(implementation, known_effect_sales)
    result = analyzer.calculate_promotion_roi(promotion_cost=1000)
    assert result.incremental_revenue == pytest.approx(8 * 8 * known_effect_sales["on_promotion"].sum())


def test_library_matching_uses_each_string_index_control_once():
    data = pd.DataFrame(
        {"sales": [18, 20, 10, 12], "promotion_applied": [1, 1, 0, 0], "store_traffic": [100, 100, 100, 100]},
        index=["t1", "t2", "c1", "c2"],
    )
    result = PromotionCausalAnalyzer(data).matching_analysis(caliper=1)
    assert result["estimated_ATE"] == pytest.approx(8)


def test_custom_treatment_is_not_inferred_as_its_own_confounder(known_effect_sales):
    analyzer = PromotionCausalAnalyzer(
        known_effect_sales.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
    )
    analyzer._define_causal_graph(treatment="store_traffic", outcome="sales")
    assert "store_traffic" not in analyzer.common_causes


def test_optional_dowhy_recovers_known_effect(known_effect_sales):
    analyzer = PromotionCausalAnalyzer(
        known_effect_sales.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
    )
    assert analyzer.estimate_ate_dowhy() == pytest.approx(8, abs=0.01)


@pytest.mark.parametrize(
    "provider,method", [("econml", "estimate_ate_causalforest"), ("doubleml", "estimate_ate_doubleml_irm")]
)
def test_optional_estimator_recovers_known_effect(provider, method, known_effect_sales):
    analyzer = PromotionCausalAnalyzer(
        known_effect_sales.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
    )
    assert getattr(analyzer, method)() == pytest.approx(8, abs=1.0)


@pytest.mark.parametrize("implementation", ["book"], indirect=True)
@pytest.mark.parametrize(
    "provider,method,effect_name",
    [
        ("econml", "double_ml_forest", "average_treatment_effect"),
        ("dowhy", "dowhy_analysis", "regression_effect"),
    ],
)
def test_optional_book_estimator_recovers_known_effect(
    implementation, known_effect_sales, provider, method, effect_name
):
    result = getattr(analyze(implementation, known_effect_sales), method)()
    assert getattr(result, effect_name) == pytest.approx(8, abs=1.0)
    if method == "dowhy_analysis":
        assert result.matching_effect == pytest.approx(8, abs=1.0)


def test_book_causal_forest_uses_effect_api_and_original_segments(monkeypatch, known_effect_sales):
    class Forest:
        def __init__(self, **kwargs):
            assert kwargs["discrete_treatment"] is True

        def fit(self, Y, T, X, W):
            assert "price" not in X

        def ate(self, X):
            return 8.0

        def effect(self, X):
            return np.full(len(X), 8.0)

    monkeypatch.setitem(sys.modules, "econml.dml", SimpleNamespace(CausalForestDML=Forest))
    module = runpy.run_path(str(ROOT / "demos/promotion_causal_book_demo.py"))
    analyzer = analyze((module["PromotionCausalAnalyzer"], "on_promotion", "sales_units"), known_effect_sales)
    result = analyzer.double_ml_forest()
    assert result.average_treatment_effect == 8
    assert result.heterogeneous_effects.by_category == {"Cat1": 8, "Cat2": 8}
    assert result.heterogeneous_effects.by_store_tier == {"Tier1": 8}


@pytest.mark.parametrize("method", ["estimate_ate_causalforest", "fit_causal_forest"])
def test_library_binary_forest_uses_classifier_and_encoded_controls(monkeypatch, known_effect_sales, method):
    from sklearn.base import is_classifier

    from agents import promotion_causal

    class Forest:
        def __init__(self, **kwargs):
            assert is_classifier(kwargs["model_t"])

        def fit(self, Y, T, X):
            assert "price" not in X
            assert "product_category_Cat2" in X
            assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X)

        def ate(self, X):
            return 8.0

        def ate_interval(self, X, alpha):
            return 7.0, 9.0

    monkeypatch.setattr(promotion_causal, "CausalForestDML", Forest)
    analyzer = PromotionCausalAnalyzer(
        known_effect_sales.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
    )
    result = getattr(analyzer, method)()
    assert result == 8 if method == "estimate_ate_causalforest" else isinstance(result, Forest)


def test_book_dowhy_receives_a_graph_object(monkeypatch, known_effect_sales):
    class Model:
        def __init__(self, **kwargs):
            assert isinstance(kwargs["graph"], nx.DiGraph)
            assert kwargs["graph"].has_edge("on_promotion", "sales_units")

        def identify_effect(self):
            return object()

        def estimate_effect(self, estimand, method_name):
            return SimpleNamespace(value=8)

        def refute_estimate(self, estimand, estimate, method_name):
            return "refutation"

    monkeypatch.setitem(sys.modules, "dowhy", SimpleNamespace(CausalModel=Model))
    module = runpy.run_path(str(ROOT / "demos/promotion_causal_book_demo.py"))
    analyzer = analyze((module["PromotionCausalAnalyzer"], "on_promotion", "sales_units"), known_effect_sales)
    result = analyzer.dowhy_analysis()
    assert result.regression_effect == 8
    assert result.matching_effect == 8


def test_doubleml_uses_public_imports_and_encoded_controls(monkeypatch, known_effect_sales):
    class Data:
        def __init__(self, data, y_col, d_cols, x_cols):
            assert "price" not in x_cols
            assert "product_category_Cat2" in x_cols
            assert all(pd.api.types.is_numeric_dtype(data[c]) for c in x_cols)

    class IRM:
        def __init__(self, data, ml_g, ml_m):
            self.coef = [8]
            self.summary = "fitted"

        def fit(self):
            pass

    monkeypatch.setitem(sys.modules, "doubleml", SimpleNamespace(DoubleMLData=Data, DoubleMLIRM=IRM))
    module = runpy.run_path(str(ROOT / "agents/promotion_causal.py"))
    analyzer = module["PromotionCausalAnalyzer"](
        known_effect_sales.rename(columns={"sales_units": "sales", "on_promotion": "promotion_applied"})
    )
    assert analyzer.estimate_ate_doubleml_irm() == 8
