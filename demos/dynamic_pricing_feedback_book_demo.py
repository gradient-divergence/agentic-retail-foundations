from __future__ import annotations

# region book:pricing-feedback-init
import json
import time
from datetime import datetime, timedelta

import redis
from kafka import KafkaConsumer, KafkaProducer


class DynamicPricingAgent:
    def __init__(self, product_id, initial_price, min_price, max_price):
        self.product_id = product_id
        self.current_price = initial_price
        self.min_price = min_price
        self.max_price = max_price
        self.price_elasticity = -1.5
        self.learning_rate = 0.05
        self.price_history = []
        self.demand_history = []
        self.redis_client = redis.Redis(host="localhost", port=6379)
        self.kafka_producer = KafkaProducer(
            bootstrap_servers="localhost:9092",
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self.kafka_consumer = KafkaConsumer(
            "sales-events",
            bootstrap_servers="localhost:9092",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

    # endregion book:pricing-feedback-init

    # region book:pricing-feedback-run-loop
    def run_feedback_loop(self):
        """Main feedback loop for continuous price optimization"""
        print(f"Starting dynamic pricing agent for product {self.product_id}")
        print(f"Initial price: ${self.current_price:.2f}")

        try:
            while True:
                recent_sales = self.get_recent_sales()
                new_price = self.compute_optimal_price(recent_sales)
                if abs(new_price - self.current_price) / self.current_price > 0.02:
                    self.update_price(new_price)
                self.process_sales_feedback()
                time.sleep(60)

        except KeyboardInterrupt:
            print("Pricing agent shutting down")
        finally:
            self.kafka_consumer.close()
            self.kafka_producer.close()

    # endregion book:pricing-feedback-run-loop

    # region book:pricing-feedback-get-sales
    def get_recent_sales(self):
        """Get recent sales data from Redis time-series database"""
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        start_ts = int(one_hour_ago.timestamp() * 1000)
        end_ts = int(now.timestamp() * 1000)
        try:
            sales_data = self.redis_client.execute_command(
                "TS.RANGE", f"sales:{self.product_id}:quantity", start_ts, end_ts
            )
            return [(entry[0], entry[1]) for entry in sales_data]
        except Exception as exc:
            print(f"Error retrieving sales data: {exc}")
            return []

    # endregion book:pricing-feedback-get-sales

    # region book:pricing-feedback-compute
    def compute_optimal_price(self, recent_sales):
        """Calculate optimal price based on elasticity model"""
        if not recent_sales or not self.price_history:
            return self.current_price

        quantities = [q for _, q in recent_sales]
        avg_hourly_demand = sum(quantities) / len(quantities) if quantities else 0
        self.price_history.append(self.current_price)
        self.demand_history.append(avg_hourly_demand)
        marginal_cost = self.min_price * 0.8

        if self.price_elasticity == -1.0:
            optimal_price = self.current_price
        else:
            optimal_markup = abs(1 / (1 + (1 / self.price_elasticity)))
            optimal_price = marginal_cost / optimal_markup

        optimal_price = max(min(optimal_price, self.max_price), self.min_price)
        print(f"Computed optimal price: ${optimal_price:.2f} (current: ${self.current_price:.2f})")
        return optimal_price

    # endregion book:pricing-feedback-compute

    # region book:pricing-feedback-update-price
    def update_price(self, new_price):
        """Apply the new price and publish price change event"""
        old_price = self.current_price
        self.current_price = new_price
        price_change_event = {
            "product_id": self.product_id,
            "old_price": old_price,
            "new_price": new_price,
            "timestamp": datetime.now().isoformat(),
            "reason": "elasticity_optimization",
        }

        self.kafka_producer.send("price-updates", price_change_event)
        print(f"Price updated: ${old_price:.2f} -> ${new_price:.2f}")

    # endregion book:pricing-feedback-update-price

    # region book:pricing-feedback-process-sales
    def process_sales_feedback(self):
        """Process incoming sales events to update elasticity model"""
        messages = self.kafka_consumer.poll(timeout_ms=500)

        for _topic_partition, batch in messages.items():
            for message in batch:
                sale = message.value
                if sale["product_id"] == self.product_id:
                    self.update_elasticity_model(sale)

    # endregion book:pricing-feedback-process-sales

    # region book:pricing-feedback-update-elasticity
    def update_elasticity_model(self, sale):
        """Update price elasticity estimate based on observed sales"""
        if len(self.price_history) < 2 or len(self.demand_history) < 2:
            return

        price_pct_change = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
        if price_pct_change == 0:
            return

        demand_pct_change = (self.demand_history[-1] - self.demand_history[-2]) / self.demand_history[-2]
        observed_elasticity = demand_pct_change / price_pct_change
        self.price_elasticity = (
            1 - self.learning_rate
        ) * self.price_elasticity + self.learning_rate * observed_elasticity
        print(f"Updated price elasticity: {self.price_elasticity:.4f}")


# endregion book:pricing-feedback-update-elasticity


# region book:pricing-feedback-main
if __name__ == "__main__":
    agent = DynamicPricingAgent(
        product_id="SKU123456",
        initial_price=29.99,
        min_price=19.99,
        max_price=39.99,
    )
    agent.run_feedback_loop()

# endregion book:pricing-feedback-main
