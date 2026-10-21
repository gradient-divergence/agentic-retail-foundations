"""Chapter 1's fixed-rule inventory loop with immediate simulated receipt."""


# region book:intro-inventory-agent
class InventoryAgent:
    def __init__(self, reorder_threshold, max_capacity):
        self.reorder_threshold = reorder_threshold
        self.max_capacity = max_capacity
        self.current_stock = 0

    def perceive(self, external_data):
        self.current_stock = external_data.get("stock_level", self.current_stock)

    def decide(self):
        if self.current_stock < self.reorder_threshold:
            order_quantity = self.max_capacity - self.current_stock
            return {"action": "reorder", "amount": order_quantity}
        return {"action": "wait"}

    def act(self, decision):
        if decision["action"] == "reorder":
            amount = decision["amount"]
            print(f"Placing order for {amount} units.")
            self.current_stock += amount

    def learn(self, feedback):
        pass


agent = InventoryAgent(reorder_threshold=50, max_capacity=100)
environment = {"stock_level": 60}

for day in range(1, 8):
    print(f"\nDay {day}: Stock level = {environment['stock_level']}")
    agent.perceive(environment)
    decision = agent.decide()
    agent.act(decision)
    sales = 15 if day == 3 else 5
    environment["stock_level"] = max(agent.current_stock - sales, 0)
    agent.learn(feedback={"sales": sales, "stock_level": environment["stock_level"]})
# endregion book:intro-inventory-agent
