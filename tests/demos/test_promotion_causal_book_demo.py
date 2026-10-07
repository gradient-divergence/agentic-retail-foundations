import runpy
from pathlib import Path

import pandas as pd
import pytest


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
