import runpy


def test_module_runs_the_recommendation_demo(capsys):
    runpy.run_path("demos/recommendation.py", run_name="__main__")
    assert "Generating recommendations for customer C1..." in capsys.readouterr().out
