import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def agent(monkeypatch):
    redis, kafka = ModuleType("redis"), ModuleType("kafka")
    redis.Redis = MagicMock()
    kafka.KafkaProducer, kafka.KafkaConsumer = MagicMock(), MagicMock()
    monkeypatch.setitem(sys.modules, "redis", redis)
    monkeypatch.setitem(sys.modules, "kafka", kafka)
    path = Path(__file__).resolve().parents[2] / "demos/dynamic_pricing_feedback_book_demo.py"
    spec = importlib.util.spec_from_file_location("pricing_book_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DynamicPricingAgent("sku", 10, 1, 100)


def test_redis_sales_samples_are_numeric(agent):
    agent.redis_client.execute_command.return_value = [[1000, b"2"], [2000, b"3.5"]]
    assert agent.get_recent_sales() == [(1000, 2.0), (2000, 3.5)]


def test_first_sample_starts_price_and_demand_history(agent):
    assert agent.compute_optimal_price([(1000, 10)]) == 10
    assert agent.price_history == [10]
    assert agent.demand_history == [10]


def test_constant_elasticity_markup_multiplies_marginal_cost(agent):
    agent.price_history, agent.demand_history = [10], [10]
    assert agent.compute_optimal_price([(1000, 10)]) == pytest.approx(2.4)


def test_zero_previous_demand_does_not_divide_by_zero(agent):
    agent.price_history, agent.demand_history = [10, 9], [0, 1]
    agent.update_elasticity_model({"product_id": "sku"})
    assert agent.price_elasticity == -1.5


def test_one_observation_pair_updates_elasticity_once_per_poll(agent):
    agent.price_history, agent.demand_history = [10, 11], [100, 80]
    agent.kafka_consumer.poll.return_value = {
        "partition": [SimpleNamespace(value={"product_id": "sku"}) for _ in range(3)]
    }
    agent.process_sales_feedback()
    assert agent.price_elasticity == pytest.approx(-1.525)
