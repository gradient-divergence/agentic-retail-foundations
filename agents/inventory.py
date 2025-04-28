"""
Inventory agent module for agentic-retail-foundations.
Defines the InventoryAgent class for inventory management using (s, S) policy.
"""


class InventoryAgent:
    """
    Agent for inventory management using an (s, S) policy.
    - s = reorder_threshold: reorder when inventory falls below this level
    - S = max_capacity: order up to this level when reordering
    """

    def __init__(self, reorder_threshold, max_capacity):
        self.reorder_threshold = (
            reorder_threshold  # When stock falls below this, agent should reorder
        )
        self.max_capacity = max_capacity  # Max storage capacity or desired stock level
        self.current_stock = 0

    def perceive(self, external_data):
        """Sense the environment: get current stock level (and any other signals)."""
        self.current_stock = external_data.get("stock_level", self.current_stock)

    def decide(self):
        """
        Reason about whether and how much to reorder.
        Implements optimal (s,S) inventory policy where:
        - s = reorder_threshold: reorder when inventory falls below this level
        - S = max_capacity: order up to this level when reordering
        Optimality condition: s and S minimize total expected cost:
        C(s,S) = ordering costs + holding costs + stockout costs
        """
        if self.current_stock < self.reorder_threshold:
            # Plan action: calculate reorder quantity up to max capacity
            order_quantity = self.max_capacity - self.current_stock
            return {"action": "reorder", "amount": order_quantity}
        else:
            # No action needed
            return {"action": "wait"}

    def act(self, decision):
        """Execute the decided action (e.g., place an order)."""
        if decision["action"] == "reorder":
            amount = decision["amount"]
            print(
                f"Placing order for {amount} units."
            )  # In real system, call supplier API
            # For simulation, assume order immediately refills stock:
            self.current_stock += amount

    def learn(self, feedback):
        """Update agent's strategy based on outcomes (simplified as no-op here)."""
        # In a real agent, you might adjust thresholds or models based on feedback.
        pass
