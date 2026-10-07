import pytest


def test_component_weights_match_published_rubric():
    import re
    from pathlib import Path

    from evals.scoring import score_outcome

    rubric = Path("evals/scoring_rubric.md").read_text()
    weights = [float(weight) for weight in re.findall(r": (0\.\d+) weight", rubric)]
    assert len(weights) == 4
    for index, weight in enumerate(weights):
        components = [0.0] * 4
        components[index] = 1
        assert score_outcome(*components)[0] == pytest.approx(weight)


def test_weighted_score_and_policy_hard_failure():
    from evals.scoring import score_outcome

    assert score_outcome(1, 1, 1, 1) == (1, True)
    score, passed = score_outcome(1, 1, 0.75, 0)
    assert score == pytest.approx(0.85)
    assert passed
    assert score_outcome(1, 0.99, 1, 1)[1] is False
    assert score_outcome(0, 1, 1, 1) == (0.5, False)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_score_rejected(value):
    from evals.scoring import score_outcome

    with pytest.raises(ValueError):
        score_outcome(value, 1, 1, 1)
