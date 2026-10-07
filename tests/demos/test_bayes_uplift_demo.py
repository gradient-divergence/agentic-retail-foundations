from pathlib import Path

import pytest
from pydantic import ValidationError

from demos.bayes_uplift_demo import AlphaBeta, PromoOutcome, bayesian_update, estimate_uplift


def test_conjugate_posterior_and_absolute_uplift():
    prior = AlphaBeta(alpha=2, beta=8)
    control, treatment = PromoOutcome(conversions=8, trials=100), PromoOutcome(conversions=11, trials=100)
    posterior = bayesian_update(prior, treatment)
    assert (posterior.alpha, posterior.beta) == (13, 97)
    assert posterior.mean() == pytest.approx(13 / 110)
    assert estimate_uplift(control, treatment) == pytest.approx(0.03)
    assert prior == AlphaBeta(alpha=2, beta=8)


def test_conversions_cannot_exceed_trials():
    with pytest.raises(ValidationError):
        PromoOutcome(conversions=2, trials=1)


def test_printed_region_runs_with_its_own_imports(capsys):
    source = Path("demos/bayes_uplift_demo.py").read_text()
    snippet = source.split("# region book:bayes-uplift-demo\n")[1].split(
        "# endregion book:bayes-uplift-demo"
    )[0]
    exec(compile(snippet, "book:bayes-uplift-demo", "exec"), {"__name__": "__main__"})
    output = capsys.readouterr().out
    assert "Posterior mean conversion rate:" in output
    assert "Estimated uplift:" in output
