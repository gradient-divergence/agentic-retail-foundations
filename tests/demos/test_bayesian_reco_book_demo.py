from demos.bayesian_reco_book_demo import (
    BayesianRecommendationAgent,
    ProductCatalogItem,
    demonstrate_bayesian_recommendations,
)


def test_recommendation_demo_prints_both_customers(monkeypatch, capsys):
    monkeypatch.setattr(BayesianRecommendationAgent, "visualize_customer_preferences", lambda *_: None)
    demonstrate_bayesian_recommendations()
    output = capsys.readouterr().out
    assert "Top 5 recommendations for customer C1:" in output
    assert "Generating recommendations for new customer C2..." in output
    assert output.count(" -> ") == 10


def test_category_prior_gets_conjugate_positive_and_negative_updates():
    agent = BayesianRecommendationAgent({"A": ProductCatalogItem(name="Tee", category="apparel")})
    agent.category_affinity = {"customer": {"apparel": 0.8}}
    for interaction in [True, True, False]:
        agent.update_preference("customer", "A", interaction)
    assert agent.customer_preferences["customer"]["A"] == {"alpha": 6, "beta": 2, "interactions": 3}
    assert agent.explain_recommendation("customer", "A").expected_preference == 0.75
