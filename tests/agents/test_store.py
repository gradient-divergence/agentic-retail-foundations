"""
Tests for the StoreAgent class.
"""

from dataclasses import replace
from unittest.mock import AsyncMock, patch

import pytest

from agents.store import StoreAgent
from models.task import Bid, Task, TaskStatus, TaskType


# Fixtures
@pytest.fixture
def sample_task() -> Task:
    """Provides a sample task."""
    return Task(
        type=TaskType.DELIVERY,
        description="Test Task",
        urgency=5,
        required_capacity=2,
        location="North",
    )


@pytest.fixture
def high_capacity_task() -> Task:
    """Provides a task requiring high capacity."""
    return Task(
        type=TaskType.RESTOCKING,
        description="High Capacity Task",
        urgency=8,
        required_capacity=15,
        location="South",
    )


@pytest.fixture
def store_agent_north() -> StoreAgent:
    """Provides a sample StoreAgent."""
    return StoreAgent(agent_id="S_NORTH", name="North", capacity=10, efficiency=1.0)


@pytest.fixture
def store_agent_south_busy() -> StoreAgent:
    """Provides a sample StoreAgent that is already partially busy."""
    agent = StoreAgent(agent_id="S_SOUTH", name="South", capacity=20, efficiency=1.2)
    # Add a task that uses some capacity
    busy_task = Task(
        type=TaskType.INVENTORY_CHECK,
        description="Ongoing task",
        urgency=3,
        required_capacity=8,
        status=TaskStatus.IN_PROGRESS,
    )
    agent.assigned_tasks = [busy_task]
    return agent


# Tests
def test_store_agent_initialization(store_agent_north):
    """Test basic initialization and post_init behavior."""
    assert store_agent_north.agent_id == "S_NORTH"
    assert store_agent_north.name == "North"
    assert store_agent_north.capacity == 10
    assert store_agent_north.efficiency == 1.0
    assert store_agent_north.location == "North"  # Set by post_init
    assert store_agent_north.assigned_tasks == []


def test_calculate_bid_sufficient_capacity(store_agent_north, sample_task):
    """Test bid calculation when agent has enough capacity."""
    bid = store_agent_north.calculate_bid(sample_task)
    assert isinstance(bid, Bid)
    assert bid.agent_id == store_agent_north.agent_id
    assert bid.task_id == sample_task.id

    # --- Verify calculation --- #
    # base_cost = 2 * 2.0 = 4.0
    # efficiency_cost = 1.0
    # urgency_factor = 1 + (5 / 20.0) * (0 / 10) = 1.0
    # location_penalty = 0.0 (task location = North, agent location = North)
    # expected_bid = (4.0 * 1.0 * 1.0) + 0.0 = 4.0
    assert bid.bid_value == pytest.approx(4.0)
    # completion_time = 0 (current workload) + 2 * 1.0 (task duration) = 2.0
    assert bid.estimated_completion_time == pytest.approx(2.0)
    # -------------------------- #

    # Check capacity calculation
    expected_available = store_agent_north.capacity - 0  # No tasks assigned initially
    assert bid.agent_capacity_available == expected_available


def test_calculate_bid_insufficient_capacity(store_agent_north, high_capacity_task):
    """Test bid calculation returns None when capacity is exceeded."""
    # Task requires 15, agent has 10
    bid = store_agent_north.calculate_bid(high_capacity_task)
    assert bid is None


def test_calculate_bid_busy_agent(store_agent_south_busy, sample_task):
    """Test bid calculation considers existing assigned tasks."""
    # Agent S_SOUTH: capacity=20, efficiency=1.2, used=8
    # Sample Task: capacity=2, urgency=5
    adjusted_task = replace(sample_task, location="South")  # Match location
    bid = store_agent_south_busy.calculate_bid(adjusted_task)
    assert isinstance(bid, Bid)
    expected_available = store_agent_south_busy.capacity - 8  # 8 used by busy_task
    assert bid.agent_capacity_available == expected_available

    # --- Verify calculation --- #
    # base_cost = 2 * 2.0 = 4.0
    # efficiency_cost = 1.2
    # urgency_factor = 1 + (5 / 20.0) * (1 / 20) = 1 + 0.25 * 0.05 = 1.0125
    # location_penalty = 0.0
    # expected_bid = (4.0 * 1.2 * 1.0125) + 0.0 = 4.86
    assert bid.bid_value == pytest.approx(4.86)
    # completion_time = (8 * 1.2) (current workload) + (2 * 1.2) (task duration)
    # = 9.6 + 2.4 = 12.0
    assert bid.estimated_completion_time == pytest.approx(12.0)
    # -------------------------- #


def test_calculate_bid_busy_agent_insufficient(store_agent_south_busy, high_capacity_task):
    """Test bid calculation fails if busy agent doesn't have enough remaining
    capacity."""
    # Agent has capacity 20, used 8, needs 15 -> requires 23 total -> insufficient
    adjusted_task = replace(high_capacity_task, location="South")  # Match location
    bid = store_agent_south_busy.calculate_bid(adjusted_task)
    assert bid is None


def test_calculate_bid_location_mismatch(store_agent_north, sample_task):
    """Test that location mismatch increases bid value (penalty)."""
    task_north = replace(sample_task, location="North")
    task_south = replace(sample_task, location="South")

    bid_north = store_agent_north.calculate_bid(task_north)
    bid_south = store_agent_north.calculate_bid(task_south)

    assert isinstance(bid_north, Bid)
    assert isinstance(bid_south, Bid)

    # --- Verify calculation difference --- #
    # bid_north = 4.0 (calculated above)
    # bid_south = 4.0 (base) + 5.0 (location penalty) = 9.0
    assert bid_north.bid_value == pytest.approx(4.0)
    assert bid_south.bid_value == pytest.approx(9.0)
    # ----------------------------------- #
    assert bid_south.bid_value > bid_north.bid_value  # Bid for South task should be higher due to penalty


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_execute_task_success(mock_sleep, store_agent_north, sample_task, monkeypatch):
    """Test successful task execution flow and sleep duration."""
    # Patch random.random to ensure success
    monkeypatch.setattr("random.random", lambda: 0.1)  # Guarantees < success_probability
    store_agent_north.assigned_tasks.append(sample_task)  # Assume task was assigned

    assert sample_task.status == TaskStatus.PENDING  # Initial state
    success = await store_agent_north.execute_task(sample_task)

    assert success is True
    assert sample_task.status == TaskStatus.COMPLETED
    assert sample_task not in store_agent_north.assigned_tasks  # Task removed

    # Verify sleep was called with expected duration
    # execution_time = max(0.1, task.required_capacity * self.efficiency * 0.1)
    # execution_time = max(0.1, 2 * 1.0 * 0.1) = 0.2
    mock_sleep.assert_awaited_once_with(pytest.approx(0.2))


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_execute_task_failure(mock_sleep, store_agent_north, sample_task, monkeypatch):
    """Test failed task execution flow and sleep duration."""
    # Patch random.random to ensure failure
    monkeypatch.setattr("random.random", lambda: 0.99)  # Guarantees >= success_probability
    store_agent_north.assigned_tasks.append(sample_task)

    assert sample_task.status == TaskStatus.PENDING  # Initial state
    success = await store_agent_north.execute_task(sample_task)

    assert success is False
    assert sample_task.status == TaskStatus.FAILED
    assert sample_task not in store_agent_north.assigned_tasks  # Task removed

    # Verify sleep was called with expected duration (same as success case)
    mock_sleep.assert_awaited_once_with(pytest.approx(0.2))
