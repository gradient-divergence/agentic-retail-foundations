from agents.bayesian import BayesianRecommendationAgent


def demo_bayesian_recommendations():
    # Create a simple product catalog
    product_catalog = {
        "P1": {"name": "Casual T-Shirt", "category": "apparel", "price": 19.99},
        "P2": {"name": "Running Shoes", "category": "footwear", "price": 89.99},
        "P3": {"name": "Yoga Mat", "category": "fitness", "price": 29.99},
        "P4": {"name": "Water Bottle", "category": "accessories", "price": 12.99},
        "P5": {"name": "Fitness Tracker", "category": "electronics", "price": 99.99},
        "P6": {"name": "Dumbbell Set", "category": "fitness", "price": 149.99},
        "P7": {"name": "Wireless Earbuds", "category": "electronics", "price": 79.99},
        "P8": {"name": "Backpack", "category": "accessories", "price": 49.99},
        "P9": {"name": "Athletic Shorts", "category": "apparel", "price": 29.99},
        "P10": {"name": "Protein Powder", "category": "nutrition", "price": 39.99},
    }

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

    # Simulate interactions for customer C1
    print("\nSimulating customer interactions...")
    agent.update_preference("C1", "P3", True)  # Likes yoga mat
    agent.update_preference("C1", "P3", True)  # Continues to like
    agent.update_preference("C1", "P6", True)  # Likes dumbbell set
    agent.update_preference("C1", "P10", True)  # Likes protein powder
    agent.update_preference("C1", "P1", True)  # Mixed for T-shirt
    agent.update_preference("C1", "P1", False)  # Then a negative signal
    agent.update_preference("C1", "P4", True)  # Likes water bottle
    agent.update_preference("C1", "P5", False)  # Dislikes fitness tracker
    agent.update_preference("C1", "P7", False)  # Dislikes earbuds

    # Generate recommendations for customer C1
    print("\nGenerating recommendations for customer C1...")
    all_products = list(product_catalog.keys())
    recommendations_c1 = agent.recommend("C1", all_products, num_recommendations=5)

    return recommendations_c1, agent, all_products, product_catalog
