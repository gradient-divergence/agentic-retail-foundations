"""Short demo for the OODA pricing agent."""

# region book:ooda-pricing-demo-short
import logging

from agents.ooda import OODAPricingAgent
from models.pricing import PricingProduct

logging.basicConfig(level=logging.INFO)


def demo_ooda_pricing() -> None:
    agent = OODAPricingAgent()
    product = PricingProduct(
        product_id="SKU-1001",
        name="Classic Tee",
        category="Apparel",
        cost=12.0,
        current_price=29.0,
        min_price=19.0,
        max_price=39.0,
        inventory=18,
        sales_last_7_days=[4, 5, 6, 5, 4, 3, 5],
    )
    agent.update_products({product.product_id: product})

    agent.run_cycle_for_product(product.product_id)
    updated_price = agent.products[product.product_id].current_price
    print(f"Updated price: ${updated_price:.2f}")


if __name__ == "__main__":
    demo_ooda_pricing()

# endregion book:ooda-pricing-demo-short
