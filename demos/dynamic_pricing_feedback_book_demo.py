from __future__ import annotations

# region book:pricing-feedback-init
import json
import time
from datetime import datetime
from math import isfinite, log

try:
    import redis
    from kafka import KafkaConsumer, KafkaProducer
except ModuleNotFoundError:
    raise SystemExit("Install the streaming extra: uv sync --extra streaming") from None


class DynamicPricingAgent:
    def __init__(self, product_id, initial_price, min_price, max_price, marginal_cost):
        if not isfinite(marginal_cost) or marginal_cost < 0:
            raise ValueError("Illustrative marginal cost must be finite and nonnegative")
        self.marginal_cost = marginal_cost
        self.observation_ids = []
        self.last_learned_pair = None
        self.last_price_change_ms = None
        self.window_start_ms = None
        self.window_end_ms = None
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
        # Closed, non-overlapping UTC epoch hours give stable observation identities.
        end_ts = (int(time.time() * 1000) // 3_600_000) * 3_600_000
        start_ts = end_ts - 3_600_000
        self.window_end_ms = end_ts
        self.window_start_ms = start_ts
        try:
            sales_data = self.redis_client.execute_command(
                "TS.RANGE", f"sales:{self.product_id}:quantity", start_ts, end_ts - 1
            )
            return [(int(entry[0]), float(entry[1])) for entry in sales_data]
        except Exception as exc:
            print(f"Error retrieving sales data: {exc}")
            self.last_learned_pair = tuple(self.observation_ids[-2:])
            return None

    # endregion book:pricing-feedback-get-sales

    def _validated_sales_window(self, recent_sales):
        if recent_sales is None:
            return None
        if (
            self.last_price_change_ms is not None
            and self.window_start_ms is not None
            and self.last_price_change_ms > self.window_start_ms
        ):
            return None  # Mixed-price windows cannot be attributed to one price.
        identity = (self.product_id, self.window_start_ms, self.window_end_ms)
        if identity in self.observation_ids:
            return None
        quantities = [q for _, q in recent_sales]
        if any(not isfinite(q) or q < 0 for q in quantities):
            raise ValueError("Per-sale quantities must be finite and nonnegative")
        return identity, quantities

    def _record_observation(self, identity, demand):
        self.observation_ids.append(identity)
        self.price_history.append(self.current_price)
        self.demand_history.append(demand)
        self.price_history = self.price_history[-120:]
        self.demand_history = self.demand_history[-120:]
        self.observation_ids = self.observation_ids[-120:]

    # region book:pricing-feedback-compute
    def compute_optimal_price(self, recent_sales):
        """Calculate optimal price based on elasticity model"""
        # Full helpers: demos/dynamic_pricing_feedback_book_demo.py.
        # _validated_sales_window checks replay, mixed-price windows and quantities.
        # _record_observation retains the bounded price/demand history.
        observation = self._validated_sales_window(recent_sales)
        if observation is None:
            return self.current_price
        identity, quantities = observation
        # Redis samples are transaction quantities in the queried one-hour window, including zero sales.
        avg_hourly_demand = sum(quantities)
        self._record_observation(identity, avg_hourly_demand)
        if len(self.price_history) < 2 or avg_hourly_demand == 0:
            return self.current_price
        marginal_cost = self.marginal_cost

        if self.price_elasticity >= -1.0:
            optimal_price = self.current_price
        else:
            optimal_markup = abs(1 / (1 + (1 / self.price_elasticity)))
            optimal_price = marginal_cost * optimal_markup

        optimal_price = max(min(optimal_price, self.max_price), self.min_price)
        print(f"Computed optimal price: ${optimal_price:.2f} (current: ${self.current_price:.2f})")
        return optimal_price

    # endregion book:pricing-feedback-compute

    # region book:pricing-feedback-update-price
    def update_price(self, new_price):
        """Apply the new price and publish price change event"""
        old_price = self.current_price
        self.current_price = new_price
        self.last_price_change_ms = int(time.time() * 1000)
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
                    return  # One update per observation pair, regardless of batch size.

    # endregion book:pricing-feedback-process-sales

    # region book:pricing-feedback-update-elasticity
    def update_elasticity_model(self, sale):
        """Log-ratio elasticity assumes positive, uncensored demand and constant demand scale."""
        pair = tuple(self.observation_ids[-2:])
        if len(pair) != 2 or pair == self.last_learned_pair:
            return
        self.last_learned_pair = pair
        if len(self.price_history) < 2 or len(self.demand_history) < 2:
            return
        if any(value <= 0 for value in self.price_history[-2:] + self.demand_history[-2:]):
            return

        log_price_change = log(self.price_history[-1] / self.price_history[-2])
        if log_price_change == 0:
            return

        log_demand_change = log(self.demand_history[-1] / self.demand_history[-2])
        observed_elasticity = log_demand_change / log_price_change
        if not isfinite(observed_elasticity):
            return
        self.price_elasticity = (
            1 - self.learning_rate
        ) * self.price_elasticity + self.learning_rate * observed_elasticity
        print(f"Updated price elasticity: {self.price_elasticity:.4f}")


# endregion book:pricing-feedback-update-elasticity


# region book:pricing-feedback-main
if __name__ == "__main__":
    import socket

    try:
        for port in (6379, 9092):
            with socket.create_connection(("localhost", port), timeout=1):
                pass
    except OSError:
        raise SystemExit("Start Redis on localhost:6379 and Kafka on localhost:9092 for this demo.") from None

    agent = DynamicPricingAgent(
        product_id="SKU123456",
        initial_price=29.99,
        min_price=19.99,
        max_price=39.99,
        marginal_cost=15.0,  # Explicit illustrative cost; independent of the price floor.
    )
    agent.run_feedback_loop()

# endregion book:pricing-feedback-main
