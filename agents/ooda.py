"""
OODA (Observe-Orient-Decide-Act) agent for dynamic pricing in agentic-retail-foundations.
"""

from datetime import datetime
import random
import logging
from models.pricing import PricingProduct

logger = logging.getLogger("AgentFrameworks")


class OODAPricingAgent:
    """
    An agent for dynamic pricing using the OODA loop.
    Implements Observe, Orient, Decide, and Act phases for retail pricing.
    """

    def __init__(
        self,
        inventory_weight=0.3,
        competitor_weight=0.4,
        sales_weight=0.3,
        max_price_change_pct=5.0,
    ):
        self.products: dict[str, PricingProduct] = {}
        self.inventory_weight = inventory_weight
        self.competitor_weight = competitor_weight
        self.sales_weight = sales_weight
        self.max_price_change_pct = max_price_change_pct
        self.action_history: list[dict] = []

        logger.info(
            f"OODA Pricing Agent init (Inv={inventory_weight}, "
            f"Comp={competitor_weight}, Sales={sales_weight}, "
            f"MaxChange={max_price_change_pct}%)"
        )

    def update_products(self, products_data: dict[str, PricingProduct]):
        self.products = products_data
        logger.info(f"Updated agent with {len(products_data)} products.")

    def observe(self, product_id: str) -> dict:
        if product_id not in self.products:
            logger.warning(f"Observe: {product_id} not found.")
            return {}
        product = self.products[product_id]

        competitor_prices = self._fetch_competitor_prices(product)
        inventory = product.inventory
        sales_last_7d = product.sales_last_7_days

        product.competitor_prices = competitor_prices
        product.inventory = inventory
        product.sales_last_7_days = sales_last_7d

        observation = {
            "timestamp": datetime.now(),
            "product_id": product_id,
            "current_price": product.current_price,
            "cost": product.cost,
            "inventory": inventory,
            "competitor_prices": competitor_prices,
            "sales_last_7_days": sales_last_7d,
        }
        return observation

    def orient(self, product_id: str, observation: dict) -> dict:
        if not observation or product_id not in self.products:
            logger.warning(f"Orient: Missing data for {product_id}.")
            return {}

        product = self.products[product_id]
        comp_prices = observation.get("competitor_prices", {})
        if comp_prices:
            avg_comp = sum(comp_prices.values()) / len(comp_prices)
        else:
            avg_comp = product.current_price

        if product.current_price > avg_comp * 1.1:
            price_pos = "premium"
        elif product.current_price < avg_comp * 0.9:
            price_pos = "discount"
        else:
            price_pos = "competitive"

        inv = observation["inventory"]
        low_thr, high_thr = 10, 50
        if inv < low_thr:
            inv_status = "low"
        elif inv > high_thr:
            inv_status = "high"
        else:
            inv_status = "optimal"

        sales = observation.get("sales_last_7_days", [])
        avg_s = sum(sales) / len(sales) if sales else 0.0
        days_of_supply = inv / avg_s if avg_s > 0 else float("inf")

        if inv_status == "low" and avg_s > 1:
            sales_assess = "risk_of_stockout"
        elif inv_status == "high" and avg_s < 1:
            sales_assess = "slow_moving"
        elif avg_s <= 0.1 and inv > 0:
            sales_assess = "stagnant"
        else:
            sales_assess = "normal"

        if inv_status == "low" and sales_assess == "risk_of_stockout":
            situation = "high_demand_low_supply"
        elif inv_status == "high" and sales_assess in ["slow_moving", "stagnant"]:
            situation = "low_demand_high_supply"
        elif price_pos == "premium" and sales_assess in ["slow_moving", "stagnant"]:
            situation = "price_sensitive_market"
        elif price_pos == "discount" and sales_assess == "normal":
            situation = "underpriced"
        else:
            situation = "balanced"

        orientation = {
            "timestamp": datetime.now(),
            "product_id": product_id,
            "avg_competitor_price": avg_comp,
            "price_position": price_pos,
            "inventory_status": inv_status,
            "sales_assessment": sales_assess,
            "market_situation": situation,
            "avg_daily_sales_7d": avg_s,
            "days_of_supply": days_of_supply,
        }
        logger.info(
            f"Orient {product_id}: {situation} "
            f"(Inv={inv_status}, Sales={sales_assess}, Price={price_pos})"
        )
        return orientation

    def decide(self, product_id: str, orientation: dict) -> dict:
        if not orientation or product_id not in self.products:
            logger.warning(f"Decide: Missing orientation for {product_id}.")
            return {}

        product = self.products[product_id]
        curr_price = product.current_price

        # Price difference from competitor
        avg_comp = orientation.get("avg_competitor_price", curr_price)
        price_diff_pct = (curr_price - avg_comp) / avg_comp * 100 if avg_comp else 0

        inv_status = orientation.get("inventory_status")
        sales_assess = orientation.get("sales_assessment")

        # Basic percentage adjustments
        inv_component = 0.0
        comp_component = 0.0
        sales_component = 0.0

        if inv_status == "low":
            inv_component = 2.0
        elif inv_status == "high":
            inv_component = -3.0

        if abs(price_diff_pct) > 5:
            comp_component = -(price_diff_pct / 3.0)

        if sales_assess == "risk_of_stockout":
            sales_component = 2.5
        elif sales_assess == "slow_moving":
            sales_component = -2.5
        elif sales_assess == "stagnant":
            sales_component = -4.0

        total_change = (
            inv_component * self.inventory_weight
            + comp_component * self.competitor_weight
            + sales_component * self.sales_weight
        )
        capped_change = max(
            -self.max_price_change_pct, min(self.max_price_change_pct, total_change)
        )
        new_price = curr_price * (1 + capped_change / 100)

        # Respect min/max
        new_price = max(product.min_price, min(product.max_price, new_price))
        new_price = self._apply_price_psychology(new_price)

        # Identify main driver
        comps = {
            "inventory": abs(inv_component * self.inventory_weight),
            "competitor": abs(comp_component * self.competitor_weight),
            "sales": abs(sales_component * self.sales_weight),
        }
        main_driver = (
            max(comps, key=comps.get) if any(v > 0 for v in comps.values()) else "none"
        )

        decision = {
            "timestamp": datetime.now(),
            "product_id": product_id,
            "old_price": curr_price,
            "new_price": round(new_price, 2),
            "capped_change_pct": capped_change,
            "primary_driver": main_driver,
        }
        logger.info(
            f"Decide {product_id}: ${curr_price:.2f} -> ${new_price:.2f} "
            f"(Change={capped_change:.2f}%, Driver={main_driver})"
        )
        return decision

    def act(self, product_id: str, decision: dict) -> bool:
        if not decision or product_id not in self.products:
            logger.warning(f"Act: Missing decision for {product_id}.")
            return False
        product = self.products[product_id]
        old_p = decision.get("old_price", product.current_price)
        new_p = decision.get("new_price", product.current_price)

        if abs(new_p - old_p) < 0.01:
            logger.info(f"Act {product_id}: price change too small, skipping.")
            return False

        success = True  # pretend success
        if success:
            product.current_price = new_p
            action_log = {
                "timestamp": datetime.now(),
                "product_id": product_id,
                "old_price": old_p,
                "new_price": new_p,
                "reason": decision.get("primary_driver", "unknown"),
            }
            self.action_history.append(action_log)
            logger.info(f"Act {product_id}: Price updated to {new_p:.2f}")
        else:
            logger.error(f"Act {product_id}: update failed.")
        return success

    def run_cycle_for_product(self, product_id: str) -> bool:
        logger.info(f"\n--- OODA Cycle Start: {product_id} ---")
        obs = self.observe(product_id)
        if not obs:
            return False
        ori = self.orient(product_id, obs)
        if not ori:
            return False
        dec = self.decide(product_id, ori)
        if not dec:
            return False
        acted = self.act(product_id, dec)
        logger.info(f"--- OODA Cycle End: {product_id} (Action Taken: {acted}) ---")
        return acted

    # Helpers
    def _fetch_competitor_prices(self, product: PricingProduct) -> dict[str, float]:
        base_price = product.current_price
        noise_a = random.uniform(-0.08, 0.08)
        noise_b = random.uniform(-0.05, 0.10)
        comp_a = round(max(product.min_price, base_price * (1 + noise_a)), 2)
        comp_b = round(max(product.min_price, base_price * (1 + noise_b)), 2)
        return {"CompetitorA": comp_a, "CompetitorB": comp_b}

    def _apply_price_psychology(self, price: float) -> float:
        if price < 1.0:
            return round(price, 2)
        cents = price - int(price)
        if cents > 0.9:
            return float(int(price)) + 0.99
        else:
            return float(int(price)) + 0.99
