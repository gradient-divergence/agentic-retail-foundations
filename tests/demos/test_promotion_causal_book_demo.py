import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def prepared_fixture(analyzer):
    from demos.promotion_causal_book_demo import simulate_promotion_data

    data = simulate_promotion_data(seed=42)
    dates = pd.to_datetime(data["date"])
    data["day_of_week"] = dates.dt.dayofweek
    data["month"] = dates.dt.month
    data["weekend"] = dates.dt.dayofweek >= 5
    data["holiday"] = analyzer._is_holiday(dates)
    return data


def test_matching_without_eligible_controls_is_not_a_zero_effect():
    from demos.promotion_causal_book_demo import PromotionCausalAnalyzer

    analyzer = PromotionCausalAnalyzer.__new__(PromotionCausalAnalyzer)
    analyzer.analysis_data = pd.DataFrame(
        {
            "day_of_week": [0, 0],
            "month": [1, 1],
            "weekend": [False, False],
            "holiday": [False, False],
            "product_category": ["A", "A"],
            "store_tier": ["A", "A"],
            "store_traffic": [0, 100],
            "on_promotion": [False, True],
            "sales_units": [10, 20],
        }
    )
    with pytest.raises(ValueError, match="No eligible matches; effect is not estimable"):
        analyzer.matching_analysis(max_distance=0)


def test_book_example_runs_offline(tmp_path, monkeypatch, capsys):
    path = Path(__file__).resolve().parents[2] / "demos/promotion_causal_book_demo.py"
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(path), run_name="__main__")
    output = capsys.readouterr().out
    assert "Promotion ROI:" in output
    assert "Counterfactual Analysis:" in output
    assert "Error" not in output


def test_effect_modifiers_enter_the_forest_features(monkeypatch):
    import numpy as np

    from demos.promotion_causal_book_demo import PromotionCausalAnalyzer

    analyzer = PromotionCausalAnalyzer.__new__(PromotionCausalAnalyzer)
    analyzer.analysis_data = prepared_fixture(analyzer)
    captured = {}

    class Forest:
        def __init__(self, **kwargs):
            assert kwargs["random_state"] == 42

        def fit(self, y, t, X, W=None):
            captured.update(X=X, W=W)

        def ate(self, X):
            return 1.0

        def effect(self, X):
            return np.ones(len(X))

    monkeypatch.setitem(sys.modules, "econml.dml", SimpleNamespace(CausalForestDML=Forest))
    analyzer.double_ml_forest()
    assert any(column.startswith("product_category_") for column in captured["X"].columns)


def test_adjustment_design_has_no_duplicate_calendar_feature():
    import numpy as np

    from demos.promotion_causal_book_demo import PromotionCausalAnalyzer

    analyzer = PromotionCausalAnalyzer.__new__(PromotionCausalAnalyzer)
    features = analyzer._promotion_features(prepared_fixture(analyzer))
    assert np.linalg.matrix_rank(features.to_numpy()) == features.shape[1]
