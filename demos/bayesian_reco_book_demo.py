# region book:bayes-reco-imports
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel
from scipy.stats import beta


class ProductCatalogItem(BaseModel):
    name: str
    category: str


class RecommendationExplanation(BaseModel):
    explanation: str
    expected_preference: float | None = None
    confidence: float | None = None
    interactions: int | None = None


class BayesianRecommendationAgent:
    """
    A Bayesian agent for product recommendations that balances exploration
    (learning customer preferences) with exploitation (recommending products
    likely to be purchased).

    The agent models customer preferences using Beta distributions and updates
    these distributions as new interaction data arrives.
    """

    def __init__(
        self,
        product_catalog: dict[str, ProductCatalogItem],
        exploration_weight: float = 0.3,
    ):
        """
        Initialize the recommendation agent.
        """
        self.product_catalog = product_catalog
        self.exploration_weight = exploration_weight

        # Initialize preference models for all customer-product combinations
        self.customer_preferences: dict[str, dict[str, dict[str, float]]] = {}

        # Prior beliefs about customer preferences, possibly used for category-level data
        self.category_affinity: dict[str, dict[str, float]] = {}

        print(f"Bayesian Recommendation Agent initialized with {len(product_catalog)} products")

    # endregion book:bayes-reco-imports

    # region book:bayes-reco-get-prior
    def get_product_prior(self, customer_id: str, product_id: str) -> tuple[float, float]:
        """
        Determine prior parameters for the Beta distribution representing
        our initial belief about a customer's preference for a product.
        """
        product = self.product_catalog[product_id]
        category = product.category
        # endregion book:bayes-reco-get-prior

        # region book:bayes-reco-category-affinity
        # If we have category affinity data for this customer, use it for the prior
        if customer_id in self.category_affinity and category in self.category_affinity[customer_id]:
            affinity = self.category_affinity[customer_id][category]

            # Example logic:
            # - High affinity (e.g., 0.8) might map to Beta(4,1)
            # - Moderate affinity (e.g., 0.5) might map to Beta(2,2)
            # - Low affinity (e.g., 0.2) might map to Beta(1,4)
            if affinity > 0.7:
                return (4, 1)  # Strong optimism about this category
            if affinity > 0.4:
                return (2, 2)  # Balanced moderate prior
            return (1, 4)  # More skeptical prior

        # Default to a uniform Beta(1,1) if no category info is available
        return (1, 1)
        # endregion book:bayes-reco-category-affinity

    # region book:bayes-reco-update-preference
    def update_preference(self, customer_id: str, product_id: str, interaction: bool) -> None:
        """
        Update preference model based on customer interaction.
        """
        # If this is the first time we see this customer, create a new record
        if customer_id not in self.customer_preferences:
            self.customer_preferences[customer_id] = {}
        # endregion book:bayes-reco-update-preference

        # region book:bayes-reco-init-product
        # If it's the first time we see this product-customer pair, initialize with prior
        if product_id not in self.customer_preferences[customer_id]:
            alpha, beta_val = self.get_product_prior(customer_id, product_id)
            self.customer_preferences[customer_id][product_id] = {
                "alpha": alpha,
                "beta": beta_val,
                "interactions": 0,
            }

        # Retrieve current preference model
        pref = self.customer_preferences[customer_id][product_id]
        # endregion book:bayes-reco-init-product

        # region book:bayes-reco-update-beta
        # Update Beta(α, β) based on positive or negative feedback
        if interaction:
            pref["alpha"] += 1
        else:
            pref["beta"] += 1

        pref["interactions"] += 1

        # (Optional) We could also update category-level affinity here if desired
        # endregion book:bayes-reco-update-beta

    # region book:bayes-reco-recommend
    def recommend(
        self,
        customer_id: str,
        candidate_products: list[str],
        num_recommendations: int = 5,
    ) -> list[str]:
        """
        Generate personalized product recommendations using Thompson sampling,
        which balances exploitation (recommending products with high expected
        preference) with exploration (trying products with uncertain preference).
        """

        # Initialize preferences for new customers or products as needed
        if customer_id not in self.customer_preferences:
            # Initialize preferences for a new customer
            self.customer_preferences[customer_id] = {}

        product_scores = []

        for product_id in candidate_products:
            # If we've never modeled this product for this customer, create it
            if product_id not in self.customer_preferences[customer_id]:
                alpha, beta_val = self.get_product_prior(customer_id, product_id)
                self.customer_preferences[customer_id][product_id] = {
                    "alpha": alpha,
                    "beta": beta_val,
                    "interactions": 0,
                }

            # Perform Thompson sampling by drawing from Beta distributions
            # and adding exploration bonuses
            # Retrieve the Beta parameters
            pref = self.customer_preferences[customer_id][product_id]
            alpha, beta_val = pref["alpha"], pref["beta"]

            # Thompson sampling: draw a random sample from the Beta distribution
            preference_sample = np.random.beta(alpha, beta_val)

            # Provide an additional exploration bonus if the distribution is uncertain
            # Variance of Beta(α, β) is αβ / [(α+β)²(α+β+1)]
            uncertainty = (alpha * beta_val) / ((alpha + beta_val) ** 2 * (alpha + beta_val + 1))
            exploration_bonus = self.exploration_weight * uncertainty

            # Combine preference predictions with exploration bonuses to
            # determine final product scores
            # Combine the Beta sample with the exploration bonus
            score = preference_sample + exploration_bonus

            product_scores.append((product_id, score))

        # Sort by descending score and pick top products
        product_scores.sort(key=lambda x: x[1], reverse=True)
        recommended_products = [p[0] for p in product_scores[:num_recommendations]]

        return recommended_products

    # endregion book:bayes-reco-recommend

    # region book:bayes-reco-explain
    def explain_recommendation(self, customer_id: str, product_id: str) -> RecommendationExplanation:
        """
        Provide an explanation for why a product was recommended.
        """
        # endregion book:bayes-reco-explain

        # region book:bayes-reco-explain-missing
        if (
            customer_id not in self.customer_preferences
            or product_id not in self.customer_preferences[customer_id]
        ):
            return RecommendationExplanation(explanation="This product matches your general preferences.")

        # Get Beta parameters
        pref = self.customer_preferences[customer_id][product_id]
        alpha, beta_val = pref["alpha"], pref["beta"]

        # Expected preference from Beta(α, β) is α / (α + β)
        expected_preference = alpha / (alpha + beta_val)
        # endregion book:bayes-reco-explain-missing

        # region book:bayes-reco-explain-logic
        # Use (α + β) as a rough measure of how confident we are (more interactions -> more confident)
        certainty = alpha + beta_val

        if pref["interactions"] == 0:
            reason = "This product has no recorded interactions yet."
        elif expected_preference > 0.7 and certainty > 10:
            reason = "You've shown consistent enthusiasm for similar products."
        elif expected_preference > 0.6:
            reason = "You've had mostly positive reactions to products like this."
        elif certainty < 5:
            reason = "We are exploring this recommendation to learn more about your tastes."
        else:
            reason = "This item appears to match your preferences."

        return RecommendationExplanation(
            explanation=reason,
            expected_preference=expected_preference,
            confidence=min(1.0, certainty / 20),
            interactions=pref["interactions"],
        )
        # endregion book:bayes-reco-explain-logic

    # region book:bayes-reco-visualize
    def visualize_customer_preferences(self, customer_id: str, top_n: int = 10) -> None:
        """
        Visualize the preference distributions for a customer's most-interacted products.
        """
        if customer_id not in self.customer_preferences:
            print(f"No preference data for customer {customer_id}")
            return

        # Extract and sort product preferences by interaction count
        prefs = self.customer_preferences[customer_id]

        # Extract product entries sorted by descending interactions
        products = [(pid, p["interactions"], p["alpha"], p["beta"]) for pid, p in prefs.items()]
        products.sort(key=lambda x: x[1], reverse=True)
        top_products = products[:top_n]

        if not top_products:
            print(f"No product interactions for customer {customer_id}")
            return

        # Create a visualization of Beta distributions for the customer's top products
        # Plot up to 10 distributions (arranged in subplots)
        fig, axes = plt.subplots(nrows=min(len(top_products), 5), ncols=2, figsize=(12, 10))
        axes = axes.flatten()

        for i, (pid, interactions, alpha, beta_val) in enumerate(top_products):
            if i >= len(axes):
                break

            x = np.linspace(0, 1, 1000)
            y = beta.pdf(x, alpha, beta_val)  # Beta PDF

            # Complete the visualization with labels, titles, and expected preference markers
            ax = axes[i]
            ax.plot(x, y, label=f"{self.product_catalog[pid].name}")
            ax.set_xlabel("Preference")
            ax.set_ylabel("Density")

            # Show an average preference line
            expected = alpha / (alpha + beta_val)
            ax.axvline(x=expected, color="red", linestyle="--")

            ax.set_title(f"{self.product_catalog[pid].name} (Interacts: {interactions})")
            ax.legend()

        plt.tight_layout()
        plt.show()

    # endregion book:bayes-reco-visualize


# region book:bayes-reco-demo-catalog
# Example usage
def demonstrate_bayesian_recommendations():
    # Create a simple product catalog
    product_catalog = {
        "P1": ProductCatalogItem(name="Casual T-Shirt", category="apparel"),
        "P2": ProductCatalogItem(name="Running Shoes", category="footwear"),
        "P3": ProductCatalogItem(name="Yoga Mat", category="fitness"),
        "P4": ProductCatalogItem(name="Water Bottle", category="accessories"),
        "P5": ProductCatalogItem(name="Fitness Tracker", category="electronics"),
        "P6": ProductCatalogItem(name="Dumbbell Set", category="fitness"),
        "P7": ProductCatalogItem(name="Wireless Earbuds", category="electronics"),
        "P8": ProductCatalogItem(name="Backpack", category="accessories"),
        "P9": ProductCatalogItem(name="Athletic Shorts", category="apparel"),
        "P10": ProductCatalogItem(name="Protein Powder", category="nutrition"),
    }
    # endregion book:bayes-reco-demo-catalog

    # region book:bayes-reco-demo-affinity
    # Initialize the agent with some exploration weight
    agent = BayesianRecommendationAgent(product_catalog, exploration_weight=0.2)

    # Define category affinities for customer C1
    agent.category_affinity = {
        "C1": {
            "fitness": 0.8,
            "nutrition": 0.7,
            "apparel": 0.4,
            "electronics": 0.3,
            "accessories": 0.5,
            "footwear": 0.6,
        }
    }
    # endregion book:bayes-reco-demo-affinity

    # region book:bayes-reco-demo-simulate
    print("\nSimulating customer interactions...")

    # Simulate interactions for customer C1
    agent.update_preference("C1", "P3", True)  # Likes yoga mat
    agent.update_preference("C1", "P3", True)  # Continues to like
    agent.update_preference("C1", "P6", True)  # Likes dumbbell set
    agent.update_preference("C1", "P10", True)  # Likes protein powder
    agent.update_preference("C1", "P1", True)  # Mixed for T-shirt
    agent.update_preference("C1", "P1", False)  # Then a negative signal
    agent.update_preference("C1", "P4", True)  # Likes water bottle
    agent.update_preference("C1", "P5", False)  # Dislikes fitness tracker
    agent.update_preference("C1", "P7", False)  # Dislikes earbuds
    # endregion book:bayes-reco-demo-simulate

    # region book:bayes-reco-demo-recommend
    print("\nGenerating recommendations for customer C1...")
    all_products = list(product_catalog.keys())
    recommendations = agent.recommend("C1", all_products, num_recommendations=5)

    print("\nTop 5 recommendations for customer C1:")
    for i, pid in enumerate(recommendations):
        explain = agent.explain_recommendation("C1", pid)
        prod_name = product_catalog[pid].name
        reason = explain.explanation
        print(f"  {i + 1}. {prod_name} -> {reason}")
    # endregion book:bayes-reco-demo-recommend

    # region book:bayes-reco-demo-cold-start
    print("\nGenerating recommendations for new customer C2...")
    recommendations_c2 = agent.recommend("C2", all_products, num_recommendations=5)
    for i, pid in enumerate(recommendations_c2):
        explain = agent.explain_recommendation("C2", pid)
        prod_name = product_catalog[pid].name
        reason = explain.explanation
        print(f"  {i + 1}. {prod_name} -> {reason}")

    print("\nVisualizing C1's preference distributions for top products:")
    agent.visualize_customer_preferences("C1")
    # endregion book:bayes-reco-demo-cold-start


# region book:bayes-reco-main
if __name__ == "__main__":
    demonstrate_bayesian_recommendations()
# endregion book:bayes-reco-main
