import runpy

import pytest


def test_feature_contributions_explain_prediction():
    result = runpy.run_module("demos.pricing_explainability_shap_demo", run_name="__main__")
    contributions = result["shap_values"][0].sum()
    baseline = float(result["explainer"].expected_value[0])
    assert baseline + contributions == pytest.approx(result["predicted_price"])
