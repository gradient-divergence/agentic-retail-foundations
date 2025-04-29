"""
Bayesian recommendation agent for product recommendations in retail.
"""

import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger(__name__)


class BayesianRecommendationAgent:
    """
    A Bayesian agent for product recommendations that balances exploration
    (learning customer preferences) with exploitation (recommending products
    likely to be purchased).

    The agent models customer preferences using Beta distributions and updates
    these distributions as new interaction data arrives.
    """

    def __init__(
        self, product_catalog: dict[str, dict], exploration_weight: float = 0.3
    ):
        self.product_catalog = product_catalog
        self.exploration_weight = exploration_weight
        self.customer_preferences: dict[str, dict[str, dict]] = {}
        self.category_affinity: dict[str, dict[str, float]] = {}
        logger.info(
            f"Bayesian Recommendation Agent initialized with {len(product_catalog)} products"
        )

    def get_product_prior(
        self, customer_id: str, product_id: str
    ) -> tuple[float, float]:
        if product_id not in self.product_catalog:
            logger.warning(
                f"Product {product_id} not in catalog, using default prior (1,1)."
            )
            return (1.0, 1.0)
        category = self.product_catalog[product_id].get("category")
        if not category:
            logger.warning(
                f"Product {product_id} missing category, using default prior (1,1)."
            )
            return (1.0, 1.0)
        if (
            hasattr(self, "category_affinity")
            and isinstance(self.category_affinity, dict)
            and customer_id in self.category_affinity
            and isinstance(self.category_affinity[customer_id], dict)
            and category in self.category_affinity[customer_id]
        ):
            affinity = self.category_affinity[customer_id][category]
            if not isinstance(affinity, (int, float)):
                logger.warning(
                    f"Invalid affinity type ({type(affinity)}) for C:{customer_id}, Cat:{category}. Using default prior."
                )
                return (1.0, 1.0)
            logger.debug(
                f"Using category affinity {affinity:.2f} for C:{customer_id}, Cat:{category}"
            )
            if affinity > 0.7:
                prior = (4.0, 1.0)
            elif affinity > 0.4:
                prior = (2.0, 2.0)
            else:
                prior = (1.0, 4.0)
            return prior
        logger.debug(
            f"No specific category affinity found for C:{customer_id}, Cat:{category}. Using default prior (1,1)."
        )
        return (1.0, 1.0)

    def update_preference(self, customer_id: str, product_id: str, interaction: bool):
        if customer_id not in self.customer_preferences:
            self.customer_preferences[customer_id] = {}
        if product_id not in self.customer_preferences[customer_id]:
            alpha, beta_val = self.get_product_prior(customer_id, product_id)
            self.customer_preferences[customer_id][product_id] = {
                "alpha": alpha,
                "beta": beta_val,
                "interactions": 0,
            }
        pref = self.customer_preferences[customer_id][product_id]
        if interaction:
            pref["alpha"] += 1
            logger.debug(
                f"Updated C:{customer_id}, P:{product_id}: alpha -> {pref['alpha']} (Positive Interaction)"
            )
        else:
            pref["beta"] += 1
            logger.debug(
                f"Updated C:{customer_id}, P:{product_id}: beta -> {pref['beta']} (Negative Interaction)"
            )
        pref["interactions"] += 1

    def recommend(
        self,
        customer_id: str,
        candidate_products: list[str],
        num_recommendations: int = 5,
    ) -> list[str]:
        if customer_id not in self.customer_preferences:
            self.customer_preferences[customer_id] = {}
            logger.info(
                f"New customer {customer_id} encountered. Initializing preferences."
            )
        product_scores = []
        for product_id in candidate_products:
            if product_id not in self.product_catalog:
                logger.warning(
                    f"Skipping candidate product {product_id}: Not in catalog."
                )
                continue
            if product_id not in self.customer_preferences[customer_id]:
                alpha, beta_val = self.get_product_prior(customer_id, product_id)
                self.customer_preferences[customer_id][product_id] = {
                    "alpha": alpha,
                    "beta": beta_val,
                    "interactions": 0,
                }
                logger.debug(
                    f"Initialized prior for C:{customer_id}, P:{product_id}: Alpha={alpha}, Beta={beta_val}"
                )
            pref = self.customer_preferences[customer_id][product_id]
            alpha, beta_val = pref["alpha"], pref["beta"]
            epsilon = 1e-6
            safe_alpha = max(alpha, epsilon)
            safe_beta = max(beta_val, epsilon)
            try:
                preference_sample = np.random.beta(safe_alpha, safe_beta)
            except ValueError as e:
                logger.error(
                    f"Error sampling Beta({safe_alpha}, {safe_beta}) for C:{customer_id}, P:{product_id}: {e}"
                )
                preference_sample = 0.5
            denominator = (safe_alpha + safe_beta) ** 2 * (safe_alpha + safe_beta + 1)
            if denominator > epsilon:
                uncertainty = (safe_alpha * safe_beta) / denominator
            else:
                uncertainty = 0
            exploration_bonus = self.exploration_weight * uncertainty
            score = preference_sample + exploration_bonus
            logger.debug(
                f"Scoring C:{customer_id}, P:{product_id}: Sample={preference_sample:.3f}, Uncertainty={uncertainty:.3f}, Bonus={exploration_bonus:.3f}, Score={score:.3f}"
            )
            product_scores.append((product_id, score))
        product_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [p[0] for p in product_scores[:num_recommendations]]
        return recommended_products

    def explain_recommendation(self, customer_id: str, product_id: str) -> dict:
        if (
            customer_id not in self.customer_preferences
            or product_id not in self.customer_preferences[customer_id]
        ):
            if product_id in self.product_catalog:
                category = self.product_catalog[product_id].get("category")
                if (
                    category
                    and customer_id in self.category_affinity
                    and category in self.category_affinity[customer_id]
                ):
                    return {
                        "explanation": "This product aligns with categories you've shown interest in."
                    }
            return {
                "explanation": "This might be a good match based on general trends."
            }
        pref = self.customer_preferences[customer_id][product_id]
        alpha, beta_val = pref["alpha"], pref["beta"]
        if alpha + beta_val > 0:
            expected_preference = alpha / (alpha + beta_val)
        else:
            expected_preference = 0.5
        initial_prior_strength = 2
        certainty = max(0, alpha + beta_val - initial_prior_strength)
        if pref["interactions"] == 0:
            category = self.product_catalog.get(product_id, {}).get("category")
            if (
                category
                and customer_id in self.category_affinity
                and category in self.category_affinity[customer_id]
            ):
                reason = "This product aligns with categories you've shown interest in."
            else:
                reason = "We think you might like this based on general trends."
        elif expected_preference > 0.7 and certainty > 5:
            reason = "You've shown consistent enthusiasm for similar products."
        elif expected_preference > 0.6:
            reason = "You've had mostly positive reactions to products like this."
        elif certainty < 2:
            reason = (
                "We are exploring this recommendation to learn more about your tastes."
            )
        else:
            reason = "This item appears to match your preferences."
        normalized_confidence = min(1.0, certainty / 20.0)
        return {
            "explanation": reason,
            "expected_preference": round(expected_preference, 3),
            "confidence": round(normalized_confidence, 3),
            "interactions": pref["interactions"],
        }

    def visualize_customer_preferences(self, customer_id: str, top_n: int = 10):
        if customer_id not in self.customer_preferences:
            logger.warning(f"No preference data for customer {customer_id}")
            print(f"No preference data for customer {customer_id}")
            return
        prefs = self.customer_preferences[customer_id]
        products = [
            (pid, p["interactions"], p["alpha"], p["beta"])
            for pid, p in prefs.items()
            if pid in self.product_catalog
        ]
        products.sort(key=lambda x: x[1], reverse=True)
        top_products = products[:top_n]
        if not top_products:
            logger.info(f"No product interactions recorded for customer {customer_id}")
            print(f"No product interactions for customer {customer_id}")
            return
        num_plots = len(top_products)
        ncols = 2
        nrows = (num_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(12, nrows * 3.5), squeeze=False
        )
        axes = axes.flatten()
        plot_index = 0
        for pid, interactions, alpha, beta_val in top_products:
            if plot_index >= len(axes):
                break
            epsilon = 1e-6
            safe_alpha = max(alpha, epsilon)
            safe_beta = max(beta_val, epsilon)
            x = np.linspace(0, 1, 1000)
            y = beta.pdf(x, safe_alpha, safe_beta)
            ax = axes[plot_index]
            product_name = self.product_catalog[pid]["name"]
            ax.plot(x, y, label=f"Beta({alpha:.1f}, {beta_val:.1f})")
            ax.set_xlabel("Preference Probability")
            ax.set_ylabel("Density")
            if alpha + beta_val > 0:
                expected = alpha / (alpha + beta_val)
                ax.axvline(
                    x=expected,
                    color="red",
                    linestyle="--",
                    label=f"E[Pref]={expected:.2f}",
                )
            ax.set_title(f"{product_name}\n(Interactions: {interactions})")
            ax.legend()
            ax.grid(True, linestyle=":", alpha=0.6)
            plot_index += 1
        for i in range(plot_index, len(axes)):
            fig.delaxes(axes[i])
        plt.suptitle(
            f"Preference Distributions for Customer {customer_id} (Top {num_plots})",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return fig
