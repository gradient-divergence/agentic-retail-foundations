from unittest.mock import patch

from agents.bayesian import BayesianRecommendationAgent


def _make_catalog():
    return {
        "SKU_A": {"category": "shoes"},
        "SKU_B": {"category": "shoes"},
        "SKU_C": {"category": "bags"},
        "SKU_NO_CAT": {},
    }


def test_get_product_prior_defaults():
    agent = BayesianRecommendationAgent(_make_catalog())

    assert agent.get_product_prior("cust_1", "MISSING") == (1.0, 1.0)
    assert agent.get_product_prior("cust_1", "SKU_NO_CAT") == (1.0, 1.0)


def test_get_product_prior_affinity_buckets():
    agent = BayesianRecommendationAgent(_make_catalog())
    agent.category_affinity = {
        "cust_1": {"shoes": 0.8, "bags": 0.5},
        "cust_2": {"shoes": 0.2},
    }

    assert agent.get_product_prior("cust_1", "SKU_A") == (4.0, 1.0)
    assert agent.get_product_prior("cust_1", "SKU_C") == (2.0, 2.0)
    assert agent.get_product_prior("cust_2", "SKU_A") == (1.0, 4.0)


def test_update_preference_updates_counts():
    agent = BayesianRecommendationAgent(_make_catalog())

    agent.update_preference("cust_1", "SKU_A", True)
    pref = agent.customer_preferences["cust_1"]["SKU_A"]
    assert pref["alpha"] == 2.0
    assert pref["beta"] == 1.0
    assert pref["interactions"] == 1

    agent.update_preference("cust_1", "SKU_A", False)
    pref = agent.customer_preferences["cust_1"]["SKU_A"]
    assert pref["alpha"] == 2.0
    assert pref["beta"] == 2.0
    assert pref["interactions"] == 2


def test_recommend_skips_unknown_products():
    agent = BayesianRecommendationAgent(_make_catalog(), exploration_weight=0.0)
    candidates = ["SKU_A", "MISSING", "SKU_B"]

    with patch("agents.bayesian.np.random.beta", return_value=0.5):
        recs = agent.recommend("cust_1", candidates, num_recommendations=2)

    assert "MISSING" not in recs
    assert len(recs) == 2


def test_explain_recommendation_for_new_preference():
    agent = BayesianRecommendationAgent(_make_catalog())
    agent.category_affinity = {"cust_1": {"shoes": 0.9}}

    agent.customer_preferences["cust_1"] = {"SKU_A": {"alpha": 1.0, "beta": 1.0, "interactions": 0}}

    explanation = agent.explain_recommendation("cust_1", "SKU_A")
    assert "interest" in explanation["explanation"].lower()
