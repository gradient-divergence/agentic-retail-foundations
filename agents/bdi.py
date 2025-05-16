"""
BDI (Belief-Desire-Intention) agent for inventory management in agentic-retail-foundations.
"""

import logging
import random
from datetime import datetime, timedelta

from models.inventory import InventoryItem, ProductInfo, SalesData

logger = logging.getLogger("AgentFrameworks")


class InventoryBDIAgent:
    """
    A Belief-Desire-Intention agent for inventory management.
    Implements the BDI cycle: update beliefs, deliberate, generate intentions, execute intentions.
    """

    def __init__(self):
        # Beliefs
        self.products: dict[str, ProductInfo] = {}
        self.inventory: dict[str, InventoryItem] = {}
        self.sales_data: dict[str, SalesData] = {}
        self.current_date: datetime = datetime.now()
        self.store_capacity: int = 1000

        # Desires (weighted goals)
        self.goals = {
            "minimize_stockouts": 1.0,
            "minimize_excess_inventory": 0.7,
            "maximize_profit_margin": 0.5,
            "ensure_fresh_products": 0.8,
        }

        # Intentions
        self.active_intentions: list[dict] = []
        logger.info("Inventory BDI Agent initialized.")

    def update_beliefs(
        self,
        new_inventory: dict[str, InventoryItem] | None = None,
        new_sales: dict[str, SalesData] | None = None,
        new_date: datetime | None = None,
        new_products: dict[str, ProductInfo] | None = None,
    ):
        if new_products is not None:
            self.products = new_products
            for prod in self.products.values():
                if not hasattr(prod, "current_price") or prod.current_price is None:
                    prod.current_price = prod.price

        if new_inventory is not None:
            self.inventory = new_inventory
            for pid, item in self.inventory.items():
                if pid in self.products:
                    self.products[pid].inventory = item.current_stock

        if new_sales is not None:
            self.sales_data = new_sales

        if new_date is not None:
            self.current_date = new_date

        msg = f"Beliefs updated. Date: {self.current_date.date()}"
        if new_inventory:
            msg += f", InvItems: {len(new_inventory)}"
        if new_sales:
            msg += f", SalesItems: {len(new_sales)}"
        if new_products:
            msg += f", ProductItems: {len(new_products)}"
        logger.info(msg)

    def observe(self, product_id: str) -> dict:
        if product_id not in self.products or product_id not in self.inventory:
            logger.warning(f"Observe: missing product or inventory data for {product_id}.")
            return {}
        product = self.products[product_id]
        item = self.inventory[product_id]
        sales_obj = self.sales_data.get(product_id)
        competitor_prices = self._fetch_competitor_prices(product_id)
        supplier_lead_time = self._fetch_supplier_lead_time(product.supplier_id)
        product.competitor_prices = competitor_prices
        product.inventory = item.current_stock
        if sales_obj and sales_obj.daily_sales:
            product.sales_last_7_days = sales_obj.daily_sales[-7:]
        else:
            product.sales_last_7_days = []
        observation = {
            "current_price": product.current_price,
            "cost": product.cost,
            "inventory": product.inventory,
            "competitor_prices": product.competitor_prices,
            "sales_last_7_days": product.sales_last_7_days,
            "lead_time": supplier_lead_time,
            "reorder_point": item.reorder_point,
            "optimal_stock": item.optimal_stock,
        }
        return observation

    def orient(self, product_id: str, observation: dict) -> dict:
        if not observation or product_id not in self.products or product_id not in self.inventory:
            logger.warning(f"Orient: missing observation or product data for {product_id}.")
            return {}
        product = self.products[product_id]
        item = self.inventory[product_id]
        competitor_prices = observation.get("competitor_prices", {})
        if competitor_prices:
            avg_comp = sum(competitor_prices.values()) / len(competitor_prices)
        else:
            avg_comp = product.current_price
        if product.current_price > avg_comp * 1.1:
            price_position = "premium"
        elif product.current_price < avg_comp * 0.9:
            price_position = "discount"
        else:
            price_position = "competitive"
        inventory_level = observation.get("inventory", 0)
        if inventory_level < item.reorder_point:
            inventory_status = "low"
        elif inventory_level > item.optimal_stock * 1.5:
            inventory_status = "high"
        else:
            inventory_status = "optimal"
        sales = observation.get("sales_last_7_days", [])
        avg_daily_7 = sum(sales) / len(sales) if sales else 0.0
        sales_obj = self.sales_data.get(product_id)
        trend_factor = sales_obj.trend() if sales_obj else 0.0
        projected_daily = max(0, avg_daily_7 * (1 + trend_factor))
        days_of_supply = inventory_level / projected_daily if projected_daily > 0 else float("inf")
        lead_time = observation.get("lead_time", product.lead_time_days)
        buffer_days = 3
        if inventory_status == "low" and days_of_supply < (lead_time + buffer_days):
            sales_assessment = "risk_of_stockout"
        elif inventory_status == "high" and projected_daily < max(1, item.reorder_point * 0.1):
            sales_assessment = "slow_moving"
        elif avg_daily_7 <= 0.1 and inventory_level > 0:
            sales_assessment = "stagnant"
        else:
            sales_assessment = "normal"
        if inventory_status == "low" and sales_assessment == "risk_of_stockout":
            market_situation = "high_demand_low_supply"
        elif inventory_status == "high" and sales_assessment in [
            "slow_moving",
            "stagnant",
        ]:
            market_situation = "low_demand_high_supply"
        elif price_position == "premium" and sales_assessment in [
            "slow_moving",
            "stagnant",
        ]:
            market_situation = "price_sensitive_market"
        elif price_position == "discount" and sales_assessment == "normal":
            market_situation = "underpriced"
        else:
            market_situation = "balanced"
        orientation = {
            "avg_competitor_price": avg_comp,
            "price_position": price_position,
            "inventory_status": inventory_status,
            "sales_assessment": sales_assessment,
            "market_situation": market_situation,
            "days_of_supply": days_of_supply,
            "projected_daily_sales": projected_daily,
        }
        logger.info(
            f"Oriented {product_id}: {market_situation} "
            f"(Inv: {inventory_status}, Sales: {sales_assessment}, Price: {price_position}, DoS: {days_of_supply:.1f})"
        )
        return orientation

    def deliberate(self) -> list[str]:
        stockout_utility = self._evaluate_stockout_prevention() * self.goals["minimize_stockouts"]
        excess_utility = self._evaluate_excess_reduction() * self.goals["minimize_excess_inventory"]
        profit_utility = self._evaluate_profit_maximization() * self.goals["maximize_profit_margin"]
        fresh_utility = self._evaluate_freshness() * self.goals["ensure_fresh_products"]
        utilities = {
            "minimize_stockouts": stockout_utility,
            "minimize_excess_inventory": excess_utility,
            "maximize_profit_margin": profit_utility,
            "ensure_fresh_products": fresh_utility,
        }
        sorted_goals = sorted(utilities.items(), key=lambda x: x[1], reverse=True)
        prioritized_goals = [g for (g, val) in sorted_goals if val > 0.01]
        logger.info("Deliberated Goals: " + str([(g, round(utilities[g], 3)) for g in prioritized_goals]))
        return prioritized_goals

    def generate_intentions(self, prioritized_goals: list[str]) -> None:
        self.active_intentions.clear()
        processed_products: set[str] = set()
        for goal in prioritized_goals:
            if goal == "minimize_stockouts":
                self._plan_reorders(processed_products)
            elif goal == "minimize_excess_inventory":
                self._plan_inventory_reduction(processed_products)
            elif goal == "maximize_profit_margin":
                self._plan_margin_optimization(processed_products)
            elif goal == "ensure_fresh_products":
                self._plan_freshness_management(processed_products)
        logger.info(f"Generated {len(self.active_intentions)} intentions from goals: {prioritized_goals}")

    def execute_intentions(self) -> list[dict]:
        executed_actions = []
        sorted_intentions = sorted(
            self.active_intentions,
            key=lambda x: x.get("priority", 0),
            reverse=True,
        )
        processed_products_in_execution = set()
        for intention in sorted_intentions:
            pid = intention.get("product_id")
            if not pid:
                logger.warning(f"Skipping intention with no product_id: {intention}")
                continue
            if pid in processed_products_in_execution:
                continue
            action_type = intention.get("action")
            success = False
            try:
                if action_type == "reorder":
                    success = self._execute_reorder(intention)
                elif action_type == "discount":
                    success = self._execute_discount(intention)
                elif action_type == "promote":
                    success = self._execute_promotion(intention)
                elif action_type == "discount_perishable":
                    success = self._execute_perishable_discount(intention)
                else:
                    logger.warning(f"Unknown intention action type: {action_type} for {pid}")
                if success:
                    executed_actions.append(intention)
                    processed_products_in_execution.add(pid)
            except Exception as e:
                logger.error(f"Error executing intention {intention}: {e}", exc_info=True)
        logger.info(f"Executed {len(executed_actions)} intentions.")
        self.active_intentions.clear()
        return executed_actions

    def _evaluate_stockout_prevention(self) -> float:
        if not self.inventory:
            return 0.0
        at_risk_count = 0
        total_value_at_risk = 0.0
        total_value = 0.0
        for pid, item in self.inventory.items():
            if pid not in self.products or pid not in self.sales_data:
                continue
            product = self.products[pid]
            sales_obj = self.sales_data[pid]
            avg_s = sales_obj.average_daily_sales()
            tr = 1.0 + sales_obj.trend()
            proj_s = max(avg_s * tr, 0.1)
            current_val = product.price * item.current_stock
            total_value += current_val
            days_of_supply = item.current_stock / proj_s if proj_s > 0 else float("inf")
            if days_of_supply <= product.lead_time_days + 3:
                at_risk_count += 1
                total_value_at_risk += current_val
        if total_value == 0:
            return 0.0
        risk_ratio = total_value_at_risk / total_value
        sku_ratio = at_risk_count / len(self.inventory)
        return 0.4 * sku_ratio + 0.6 * risk_ratio

    def _evaluate_excess_reduction(self) -> float:
        if not self.inventory:
            return 0.0
        total_excess = 0.0
        total_cost = 0.0
        for pid, item in self.inventory.items():
            if pid not in self.products:
                continue
            product = self.products[pid]
            total_cost += item.current_stock * product.cost
            threshold = item.optimal_stock * 1.2
            if item.current_stock > threshold:
                excess_units = item.current_stock - item.optimal_stock
                total_excess += excess_units * product.cost
        if total_cost == 0:
            return 0.0
        return total_excess / total_cost

    def _evaluate_profit_maximization(self) -> float:
        return 0.1

    def _evaluate_freshness(self) -> float:
        perishable_risk = 0.0
        total_perish_cost = 0.0
        if not self.inventory:
            return 0.0
        for pid, item in self.inventory.items():
            if pid in self.products:
                prod = self.products[pid]
                if prod.shelf_life_days is not None and item.current_stock > 0:
                    total_perish_cost += item.current_stock * prod.cost
                    sales_obj = self.sales_data.get(pid)
                    if sales_obj:
                        avg_s = sales_obj.average_daily_sales()
                        tf = 1.0 + sales_obj.trend()
                        proj_s = max(avg_s * tf, 0.1)
                        days_to_sell = item.current_stock / proj_s
                        if days_to_sell > prod.shelf_life_days * 0.7:
                            perishable_risk += item.current_stock * prod.cost
                    else:
                        perishable_risk += 0.5 * (item.current_stock * prod.cost)
        if total_perish_cost == 0:
            return 0.0
        return perishable_risk / total_perish_cost

    def _plan_reorders(self, processed_products: set[str]) -> None:
        for pid, item in self.inventory.items():
            if pid in processed_products:
                continue
            if pid not in self.products or pid not in self.sales_data:
                continue
            if item.pending_order_quantity > 0 and item.expected_delivery_date and item.expected_delivery_date.date() > self.current_date.date():
                continue
            product = self.products[pid]
            sales_obj = self.sales_data[pid]
            avg_s = sales_obj.average_daily_sales()
            tr = 1.0 + sales_obj.trend()
            proj_s = max(avg_s * tr, 0.1)
            days_of_supply = item.current_stock / proj_s
            lead_time = product.lead_time_days
            buffer_days = 3
            if days_of_supply <= lead_time + buffer_days:
                needed = item.optimal_stock - (item.current_stock + item.pending_order_quantity)
                order_qty = int(round(max(needed, product.min_order_quantity)))
                if order_qty > 0:
                    urgency = 1.0 - (days_of_supply / (lead_time + buffer_days))
                    priority = max(0.0, min(1.0, urgency))
                    self.active_intentions.append(
                        {
                            "action": "reorder",
                            "product_id": pid,
                            "quantity": order_qty,
                            "supplier_id": product.supplier_id,
                            "priority": priority,
                        }
                    )
                    processed_products.add(pid)
                    logger.info(f"INTENTION (Reorder): {order_qty} x {pid} (DoS: {days_of_supply:.1f}, Prio: {priority:.2f})")

    def _plan_inventory_reduction(self, processed_products: set[str]) -> None:
        for pid, item in self.inventory.items():
            if pid in processed_products:
                continue
            if pid not in self.products:
                continue
            if any(i["product_id"] == pid for i in self.active_intentions):
                continue
            product = self.products[pid]
            if item.current_stock > item.optimal_stock * 1.5:
                excess_ratio = (item.current_stock - item.optimal_stock) / item.optimal_stock if item.optimal_stock > 0 else 2.0
                discount_pct = min(max(round(excess_ratio * 10), 5), 30)
                priority = min(1.0, excess_ratio * 0.5) * self.goals["minimize_excess_inventory"]
                self.active_intentions.append(
                    {
                        "action": "discount",
                        "product_id": pid,
                        "discount_percentage": discount_pct,
                        "priority": priority,
                    }
                )
                processed_products.add(pid)
                logger.info(
                    f"INTENTION (Discount): {discount_pct}% off {pid} (Stock: {item.current_stock}/{item.optimal_stock}, Prio: {priority:.2f})"
                )

    def _plan_margin_optimization(self, processed_products: set[str]) -> None:
        pass

    def _plan_freshness_management(self, processed_products: set[str]) -> None:
        for pid, item in self.inventory.items():
            if pid in processed_products:
                continue
            if pid not in self.products:
                continue
            product = self.products[pid]
            if product.shelf_life_days is not None and item.current_stock > 0:
                self.active_intentions.append(
                    {
                        "action": "discount_perishable",
                        "product_id": pid,
                        "quantity": item.current_stock,
                        "discount_percentage": 20,
                        "priority": 0.5 * self.goals["ensure_fresh_products"],
                    }
                )
                processed_products.add(pid)
                logger.info(f"INTENTION (Perishable Discount): 20% off {pid}")

    def _execute_reorder(self, intention: dict) -> bool:
        pid = intention["product_id"]
        qty = intention["quantity"]
        if pid not in self.inventory or pid not in self.products:
            logger.error(f"Reorder failed: {pid} not found.")
            return False
        product = self.products[pid]
        lead_time = product.lead_time_days
        delivery_date = self.current_date + timedelta(days=lead_time)
        self.inventory[pid].pending_order_quantity += qty
        self.inventory[pid].expected_delivery_date = delivery_date
        self.inventory[pid].last_reorder_date = self.current_date
        logger.info(f"EXECUTE: Reorder {qty} x {pid}. Delivery expected {delivery_date.date()}.")
        return True

    def _execute_discount(self, intention: dict) -> bool:
        pid = intention["product_id"]
        disc_pct = intention["discount_percentage"]
        if pid not in self.products:
            logger.error(f"Discount failed: {pid} not found.")
            return False
        product = self.products[pid]
        old_price = product.current_price
        new_price = round(old_price * (1 - disc_pct / 100), 2)
        product.current_price = new_price
        logger.info(f"EXECUTE: Discount {disc_pct}% for {pid}. Price {old_price:.2f} -> {new_price:.2f}")
        return True

    def _execute_promotion(self, intention: dict) -> bool:
        pid = intention["product_id"]
        logger.info(f"EXECUTE: Promotion for {pid}. (Placeholder)")
        return True

    def _execute_perishable_discount(self, intention: dict) -> bool:
        pid = intention["product_id"]
        if pid not in self.products:
            return False
        disc_pct = intention["discount_percentage"]
        product = self.products[pid]
        old_price = product.current_price
        new_price = round(old_price * (1 - disc_pct / 100), 2)
        product.current_price = new_price
        logger.info(f"EXECUTE: Perishable discount {disc_pct}% for {pid}. Price {old_price:.2f} -> {new_price:.2f}")
        return True

    def run_cycle(self) -> list[dict]:
        logger.info("\n--- Starting Agent Cycle ---")
        logger.info("Step 1: Beliefs assumed current.")
        logger.info("Step 2: Deliberating on goals...")
        goals = self.deliberate()
        if not goals:
            logger.info("No urgent goals. Cycle ends.")
            return []
        logger.info("Step 3: Generating intentions...")
        self.generate_intentions(goals)
        if not self.active_intentions:
            logger.info("No intentions generated. Cycle ends.")
            return []
        logger.info("Step 4: Executing intentions...")
        actions = self.execute_intentions()
        logger.info(f"--- Cycle complete. {len(actions)} actions executed. ---")
        return actions

    def _fetch_competitor_prices(self, product_id: str) -> dict[str, float]:
        if product_id not in self.products:
            return {}
        price = self.products[product_id].price
        noise_a = random.uniform(-0.05, 0.05)
        noise_b = random.uniform(-0.05, 0.05)
        return {
            "CompetitorA": round(max(1.0, price * (1 + noise_a)), 2),
            "CompetitorB": round(max(1.0, price * (1 + noise_b)), 2),
        }

    def _fetch_supplier_lead_time(self, supplier_id: str) -> int:
        for prod in self.products.values():
            if prod.supplier_id == supplier_id:
                return prod.lead_time_days
        logger.warning(f"No specific lead time known for {supplier_id}, using 3.")
        return 3
