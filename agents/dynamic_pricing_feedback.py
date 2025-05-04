"""
Dynamic Pricing Agent using a real-time feedback loop.

This agent continuously monitors sales data and adjusts prices based on 
estimated price elasticity.
Distinguish from agents/qlearning.py which uses Q-learning.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Optional

# Assume Redis and Kafka clients are appropriately configured/imported
import redis.asyncio as redis # Requires aioredis
from kafka import KafkaProducer, KafkaConsumer # Requires kafka-python

logger = logging.getLogger(__name__)

class DynamicPricingAgent:
    """Dynamic pricing agent using feedback loop (Extracted from notebook)"""
    
    redis_client: Optional[redis.Redis[str]] = None

    def __init__(self, product_id: str, initial_price: float, min_price: float, max_price: float, redis_host='localhost', redis_port=6379, kafka_brokers='localhost:9092'):
        self.product_id = product_id
        self.current_price = initial_price
        self.min_price = min_price
        self.max_price = max_price

        # Learning parameters
        self.price_elasticity = -1.5 # Initial estimate
        self.learning_rate = 0.05

        # Performance tracking
        self.price_history: List[Tuple[datetime, float]] = []
        self.demand_history: List[float] = []

        # Connect to data streams
        try:
             self.redis_client = redis.Redis(
                 host=redis_host, port=redis_port, decode_responses=True
             )
        except Exception as e:
             logger.error(f"Failed to connect to Redis at {redis_host}:{redis_port}: {e}")
             self.redis_client = None

        try:
            # Note: kafka-python is typically synchronous
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=kafka_brokers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                retries=3,
                request_timeout_ms=5000
            )
            self.kafka_consumer = KafkaConsumer(
                f"sales-events-{self.product_id}", # Topic per product?
                bootstrap_servers=kafka_brokers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                group_id=f"pricing-agent-{self.product_id}",
                auto_offset_reset='latest' # Process new sales
            )
        except Exception as e:
             logger.error(f"Failed to connect to Kafka at {kafka_brokers}: {e}")
             self.kafka_producer = None
             self.kafka_consumer = None

    async def run_feedback_loop(self):
        """Main feedback loop for continuous price optimization."""
        if not self.redis_client or not self.kafka_producer or not self.kafka_consumer:
             logger.error("Agent cannot run: Missing Redis or Kafka connection.")
             return
             
        logger.info(f"Starting dynamic pricing agent for product {self.product_id}")
        logger.info(f"Initial price: ${self.current_price:.2f}")

        try:
            while True:
                # 1. Observe recent sales patterns from Redis
                recent_sales = await self.get_recent_sales()
                logger.debug(f"Retrieved {len(recent_sales)} recent sales data points.")

                # 2. Compute optimal price
                new_price = self.compute_optimal_price(recent_sales)
                logger.debug(f"Computed optimal price: ${new_price:.2f}")

                # 3. Update price if sufficiently different
                if abs(new_price - self.current_price) / self.current_price > 0.02: # 2% threshold
                    await self.update_price(new_price)
                else:
                     logger.debug("Price change below threshold, maintaining current price.")

                # 4. Process feedback from actual sales (from Kafka - simplified blocking poll)
                # In a fully async system, Kafka consumption might use aiokafka
                # This is a simplified demo loop
                await self.process_sales_feedback()

                # 5. Wait a short interval before next adjustment
                logger.debug("Waiting 60s for next cycle...")
                await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info(f"Pricing agent {self.product_id} feedback loop cancelled.")
        except KeyboardInterrupt:
            logger.info(f"Pricing agent {self.product_id} stopping due to KeyboardInterrupt.")
        except Exception as e:
             logger.error(f"Error in pricing agent {self.product_id} feedback loop: {e}", exc_info=True)
        finally:
            logger.info(f"Shutting down pricing agent {self.product_id}...")
            if self.kafka_consumer: self.kafka_consumer.close()
            if self.kafka_producer: self.kafka_producer.close()
            if self.redis_client: await self.redis_client.close() 
            logger.info(f"Pricing agent {self.product_id} shut down.")

    async def get_recent_sales(self, hours_ago=1) -> List[Tuple[int, float]]:
        """Get recent sales data from Redis time-series database."""
        if not self.redis_client: return []
        
        now = datetime.now()
        start_time = now - timedelta(hours=hours_ago)
        start_ts_ms = int(start_time.timestamp() * 1000)
        end_ts_ms = int(now.timestamp() * 1000)
        key = f"sales:{self.product_id}:quantity" # Example key structure

        try:
            sales_data = await self.redis_client.execute_command(
                "TS.RANGE", key, str(start_ts_ms), str(end_ts_ms)
            )
            # Redis TS.RANGE returns list of [timestamp_str, value_str]
            return [(int(ts), float(val)) for ts, val in sales_data]
        except Exception as e:
            logger.error(f"Error retrieving sales data from Redis key '{key}': {e}")
            return []

    def compute_optimal_price(self, recent_sales: List[Tuple[int, float]]) -> float:
        """Compute optimal price based on recent sales data and elasticity."""
        if not recent_sales:
            # Ensure return is float
            return float(self.current_price)

        # Basic demand calculation (e.g., total quantity in the window)
        total_quantity = sum(quantity for _, quantity in recent_sales)
        # If using average demand per transaction/period, need number of periods/transactions
        # For simplicity, use total quantity as a proxy for recent demand level
        current_demand_proxy = total_quantity

        if not self.demand_history:
             # Ensure return is float
             return float(self.current_price)
             
        # Use very simple elasticity model: new_price = old_price * (demand_change)^(1/elasticity)
        previous_demand_proxy = self.demand_history[-1] 
        # Ensure return is float even if previous demand was 0
        if previous_demand_proxy == 0: return float(self.current_price)

        demand_change_ratio = current_demand_proxy / previous_demand_proxy
        # Ensure return is float if demand change is non-positive
        if demand_change_ratio <= 0: return float(self.current_price)
        
        try:
             price_change_ratio = (demand_change_ratio ** (1.0 / self.price_elasticity))
        except (ValueError, OverflowError):
             price_change_ratio = 1.0

        new_price = self.current_price * price_change_ratio

        # Ensure price stays within bounds
        new_price = max(self.min_price, min(self.max_price, new_price))
        logger.debug(f" Elasticity={self.price_elasticity:.2f}, D_Ratio={demand_change_ratio:.2f}, P_Ratio={price_change_ratio:.2f} -> New Price={new_price:.2f}")

        return float(round(new_price, 2))

    async def update_price(self, new_price: float):
        """Update the price, record change, publish events."""
        if not self.redis_client or not self.kafka_producer:
             logger.error("Cannot update price: Missing Redis or Kafka connection.")
             return
             
        old_price = self.current_price
        self.current_price = new_price
        self.price_history.append((datetime.now(), new_price))
        # Keep history bounded (optional)
        # if len(self.price_history) > 100: self.price_history.pop(0)

        # Publish price update event to Kafka
        event = {
            "product_id": self.product_id,
            "old_price": old_price,
            "new_price": new_price,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            self.kafka_producer.send("price-updates", event)
            self.kafka_producer.flush(timeout=1) # Try to send immediately
        except Exception as e:
            logger.error(f"Error publishing price update to Kafka: {e}")

        # Store in Redis time series for monitoring/quick lookup
        key = f"prices:{self.product_id}"
        try:
            await self.redis_client.execute_command("TS.ADD", key, "*", str(new_price))
        except Exception as e:
            logger.error(f"Error storing price in Redis TS key '{key}': {e}")

        logger.info(f"Updated price for product {self.product_id}: ${new_price:.2f} (was ${old_price:.2f})")

    async def process_sales_feedback(self, poll_timeout_ms=100):
        """Process recent sales data from Kafka to update demand history and elasticity model."""
        if not self.kafka_consumer: return
        
        # Simplified: Poll Kafka for recent messages
        # A more robust implementation might run this consumer in a separate thread/task
        # or use an async Kafka client like aiokafka.
        logger.debug("Polling Kafka for new sales events...")
        try:
            # Poll for messages with a short timeout
            # Note: kafka-python's poll is blocking
            # Using asyncio.to_thread might be better in a truly async context
            # For this demo, we accept the potential blocking call.
            messages = self.kafka_consumer.poll(timeout_ms=poll_timeout_ms)
            
            if not messages:
                logger.debug("No new sales messages received.")
                return

            total_demand_in_period = 0
            processed_count = 0
            for tp, consumer_records in messages.items():
                for record in consumer_records:
                    try:
                        sale_data = record.value # Already deserialized
                        if sale_data.get('product_id') == self.product_id:
                            quantity = sale_data.get('quantity', 0)
                            total_demand_in_period += quantity
                            processed_count += 1
                    except Exception as e:
                        logger.error(f"Error processing Kafka record: {e} - Record: {record}")
            
            logger.debug(f"Processed {processed_count} sales messages for {self.product_id}. Total demand: {total_demand_in_period}")

            if processed_count > 0:
                self.demand_history.append(total_demand_in_period)
                # Optional: Keep history bounded
                # if len(self.demand_history) > 100: self.demand_history.pop(0)
                
                # Update elasticity model if we have enough data points
                if len(self.demand_history) >= 2 and len(self.price_history) >= 2:
                    # Use the most recent demand and the price that was active *before* it
                    await self.update_elasticity_model(
                        previous_price=self.price_history[-2][1],
                        current_price=self.price_history[-1][1],
                        previous_demand=self.demand_history[-2],
                        current_demand=self.demand_history[-1]
                    )
            # Commit offsets manually if needed (depends on consumer config)
            # self.kafka_consumer.commit() 

        except Exception as e:
             logger.error(f"Error polling/processing Kafka sales messages: {e}", exc_info=True)

    async def update_elasticity_model(self, previous_price, current_price, previous_demand, current_demand):
        """Update price elasticity based on observed price and demand changes."""
        if previous_price == current_price: 
             logger.debug("Price hasn't changed, skipping elasticity update.")
             return # Price didn't change
        if previous_demand <= 0 or current_demand <= 0:
             logger.debug("Demand is zero or negative, skipping elasticity update.")
             return # Avoid division by zero or log domain issues
             
        price_change_ratio = (current_price - previous_price) / previous_price
        demand_change_ratio = (current_demand - previous_demand) / previous_demand

        if price_change_ratio == 0: # Should be caught above, but safety check
             return
             
        observed_elasticity = demand_change_ratio / price_change_ratio
        
        # Basic sanity check - elasticity should usually be negative
        if observed_elasticity > 0.1: # Allow slightly positive for noise
             logger.warning(f"Observed positive elasticity ({observed_elasticity:.2f}). Skipping update. Check data/model.")
             return
             
        # Bound elasticity to prevent extreme values
        observed_elasticity = max(-10.0, min(-0.1, observed_elasticity)) 

        # Update elasticity estimate using learning rate (Exponential Moving Average)
        self.price_elasticity = (
            (1 - self.learning_rate) * self.price_elasticity + 
            self.learning_rate * observed_elasticity
        )
        # Bound the learned elasticity as well
        self.price_elasticity = max(-10.0, min(-0.1, self.price_elasticity))

        logger.info(f"Updated elasticity for product {self.product_id}: {self.price_elasticity:.3f} (Observed: {observed_elasticity:.3f})")

        # Store updated elasticity in Redis (optional)
        key = f"elasticity:{self.product_id}"
        if self.redis_client:
            try:
                await self.redis_client.set(key, str(self.price_elasticity))
            except Exception as e:
                logger.error(f"Error storing elasticity in Redis key '{key}': {e}") 