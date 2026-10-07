import pytest

from demos.ooda_pricing_decide_excerpt import DecisionWeights, OrientationSignal, decide_price


def test_stagnant_sales_reduce_price_with_the_chapter_weight():
    decision = decide_price(20, OrientationSignal(sales_assessment="stagnant"), DecisionWeights(), 10, 40, 5)
    assert decision.new_price == pytest.approx(19.76)
    assert decision.capped_change_pct == pytest.approx(-1.2)
