import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import logging

# Module to test
from agents.task_allocation import RetailCoordinator

# Dependent models/classes (mocked or imported)
from models.task import Task, TaskStatus, TaskType, Bid
from agents.store import StoreAgent

# --- Fixtures --- #

@pytest.fixture
def coordinator() -> RetailCoordinator:
    """Provides a RetailCoordinator instance."""
    return RetailCoordinator()

@pytest.fixture
def mock_store_agent() -> MagicMock:
    """Provides a mocked StoreAgent that passes isinstance(..., StoreAgent)."""
    # Use spec=StoreAgent to make the mock pass isinstance checks
    agent = MagicMock(spec=StoreAgent, agent_id="S001", name="Mock Store 1", assigned_tasks=[])
    agent.calculate_bid = MagicMock()
    agent.execute_task = AsyncMock()
    return agent

# --- Test Initialization --- #

def test_coordinator_initialization(coordinator: RetailCoordinator):
    """Test coordinator initializes with empty agent and task lists."""
    assert coordinator.agents == {}
    assert coordinator.tasks == {}

# --- Test register_agent --- #

def test_register_agent_success(coordinator: RetailCoordinator, mock_store_agent: MagicMock):
    """Test registering a valid StoreAgent."""
    # The mock_store_agent fixture now provides a mock that passes isinstance
    coordinator.register_agent(mock_store_agent)
    assert len(coordinator.agents) == 1
    assert coordinator.agents[mock_store_agent.agent_id] is mock_store_agent

def test_register_agent_reregister(coordinator: RetailCoordinator, mock_store_agent: MagicMock, capsys):
    """Test re-registering the same agent logs a warning but updates."""
    # Ensure the new mock also has the spec
    mock_store_agent_new = MagicMock(spec=StoreAgent, agent_id=mock_store_agent.agent_id, name="Updated Mock Store")
    coordinator.register_agent(mock_store_agent)
    coordinator.register_agent(mock_store_agent_new)
    assert len(coordinator.agents) == 1
    assert coordinator.agents[mock_store_agent.agent_id] is mock_store_agent_new
    captured = capsys.readouterr()
    assert f"Warning: Re-registering agent {mock_store_agent.agent_id}" in captured.out

def test_register_agent_invalid_type(coordinator: RetailCoordinator):
    """Test registering an object that is not a StoreAgent raises TypeError."""
    invalid_agent = {"agent_id": "invalid"}
    with pytest.raises(TypeError, match="Registered entity must be a StoreAgent instance."):
        coordinator.register_agent(invalid_agent) # type: ignore

# --- Test create_task --- #

def test_create_task(coordinator: RetailCoordinator):
    """Test creating a new task."""
    task_type = TaskType.DELIVERY
    description = "Deliver urgent package"
    urgency = 9
    required_capacity = 1
    location = "Warehouse A"
    data = {"customer_id": "CUST1"}

    task_id = coordinator.create_task(
        task_type=task_type,
        description=description,
        urgency=urgency,
        required_capacity=required_capacity,
        location=location,
        data=data
    )

    assert isinstance(task_id, str)
    assert task_id in coordinator.tasks
    created_task = coordinator.tasks[task_id]

    # Verify task attributes
    assert isinstance(created_task, Task)
    assert created_task.id == task_id
    assert created_task.type == task_type
    assert created_task.description == description
    assert created_task.urgency == urgency
    assert created_task.required_capacity == required_capacity
    assert created_task.location == location
    assert created_task.data == data
    assert created_task.status == TaskStatus.ANNOUNCED # Initial status
    assert created_task.assigned_agent_id is None
    assert created_task.winning_bid is None

# --- Test allocate_task --- #

@pytest.mark.asyncio
async def test_allocate_task_success(coordinator: RetailCoordinator):
    """Test successful task allocation based on lowest bid."""
    # Use spec=StoreAgent for mocks created inline
    agent1 = MagicMock(spec=StoreAgent, agent_id="A1", name="Agent 1", assigned_tasks=[])
    agent1.calculate_bid = MagicMock()
    agent2 = MagicMock(spec=StoreAgent, agent_id="A2", name="Agent 2", assigned_tasks=[])
    agent2.calculate_bid = MagicMock()
    agent3 = MagicMock(spec=StoreAgent, agent_id="A3", name="Agent 3", assigned_tasks=[])
    agent3.calculate_bid = MagicMock()
    # No need to call mock_isinstance_func anymore
    coordinator.register_agent(agent1)
    coordinator.register_agent(agent2)
    coordinator.register_agent(agent3)

    # Create task
    task_id = coordinator.create_task(TaskType.PICKUP, "Task X", 5, 2)
    task = coordinator.tasks[task_id]

    # Mock bids
    bid1 = Bid(agent_id="A1", task_id=task_id, bid_value=10.0)
    bid2 = Bid(agent_id="A2", task_id=task_id, bid_value=8.0) # Winning bid
    bid3 = Bid(agent_id="A3", task_id=task_id, bid_value=12.0)
    agent1.calculate_bid = MagicMock(return_value=bid1)
    agent2.calculate_bid = MagicMock(return_value=bid2)
    agent3.calculate_bid = MagicMock(return_value=bid3)

    # Allocate
    winner_id = await coordinator.allocate_task(task_id)

    # Assertions
    assert winner_id == "A2"
    agent1.calculate_bid.assert_called_once_with(task)
    agent2.calculate_bid.assert_called_once_with(task)
    agent3.calculate_bid.assert_called_once_with(task)

    # Check task state
    assert task.status == TaskStatus.ALLOCATED
    assert task.assigned_agent_id == "A2"
    assert task.winning_bid == 8.0

    # Check winner agent state
    assert len(agent2.assigned_tasks) == 1
    assert agent2.assigned_tasks[0] is task
    # Check other agents
    assert len(agent1.assigned_tasks) == 0
    assert len(agent3.assigned_tasks) == 0

@pytest.mark.asyncio
async def test_allocate_task_no_bids(coordinator: RetailCoordinator):
    """Test allocation when no agents can bid."""
    # Use spec=StoreAgent for mock
    agent1 = MagicMock(spec=StoreAgent, agent_id="A1", name="Agent 1")
    agent1.calculate_bid = MagicMock(return_value=None)
    # No need to call mock_isinstance_func anymore
    coordinator.register_agent(agent1)

    # Create task
    task_id = coordinator.create_task(TaskType.PICKUP, "Task Y", 5, 2)
    task = coordinator.tasks[task_id]

    # Allocate
    winner_id = await coordinator.allocate_task(task_id)

    # Assertions
    assert winner_id is None
    agent1.calculate_bid.assert_called_once_with(task)
    assert task.status == TaskStatus.FAILED # Status updated on failure
    assert task.assigned_agent_id is None

@pytest.mark.asyncio
async def test_allocate_task_not_found(coordinator: RetailCoordinator, capsys):
    """Test allocation when task ID does not exist."""
    winner_id = await coordinator.allocate_task("INVALID_ID")
    assert winner_id is None
    captured = capsys.readouterr()
    assert "Task INVALID_ID not found" in captured.out

@pytest.mark.asyncio
async def test_allocate_task_wrong_status(coordinator: RetailCoordinator, capsys):
    """Test allocation when task is not in ANNOUNCED state."""
    # Create task (status is ANNOUNCED)
    task_id = coordinator.create_task(TaskType.PICKUP, "Task Z", 5, 2)
    task = coordinator.tasks[task_id]
    # Manually change status
    task.status = TaskStatus.ALLOCATED

    winner_id = await coordinator.allocate_task(task_id)
    assert winner_id is None
    captured = capsys.readouterr()
    assert f"Task {task_id} is not in ANNOUNCED state" in captured.out

# --- Test execute_allocated_tasks --- #

@pytest.mark.asyncio
async def test_execute_allocated_tasks_success(coordinator: RetailCoordinator):
    """Test executing successfully allocated tasks."""
    # Use spec=StoreAgent for mocks
    agent1 = MagicMock(spec=StoreAgent, agent_id="A1", name="Agent 1")
    agent1.execute_task = AsyncMock(return_value=True)
    agent2 = MagicMock(spec=StoreAgent, agent_id="A2", name="Agent 2")
    agent2.execute_task = AsyncMock(return_value=True)
    # No need to call mock_isinstance_func anymore
    coordinator.register_agent(agent1)
    coordinator.register_agent(agent2)

    # Create and allocate tasks
    task1_id = coordinator.create_task(TaskType.PICKUP, "Task 1", 5, 2)
    task2_id = coordinator.create_task(TaskType.DELIVERY, "Task 2", 6, 3)
    task3_id = coordinator.create_task(TaskType.RESTOCKING, "Task 3", 7, 4)
    # Manually allocate for simplicity in this test
    coordinator.tasks[task1_id].status = TaskStatus.ALLOCATED
    coordinator.tasks[task1_id].assigned_agent_id = "A1"
    coordinator.tasks[task2_id].status = TaskStatus.ALLOCATED
    coordinator.tasks[task2_id].assigned_agent_id = "A2"
    # Task 3 remains ANNOUNCED

    # Execute
    await coordinator.execute_allocated_tasks()

    # Assertions
    # Check execute_task was called for allocated tasks
    agent1.execute_task.assert_awaited_once_with(coordinator.tasks[task1_id])
    agent2.execute_task.assert_awaited_once_with(coordinator.tasks[task2_id])
    # Status should be updated by the mocked execute_task, but we didn't mock that side effect here.
    # In a real scenario, the agent's execute_task would change the status.

@pytest.mark.asyncio
async def test_execute_allocated_tasks_failure_and_exception(coordinator: RetailCoordinator, caplog, capsys):
    """Test handling task execution failure and exceptions."""
    # Use spec=StoreAgent for mocks
    agent1 = MagicMock(spec=StoreAgent, agent_id="A1", name="Agent 1")
    agent1.execute_task = AsyncMock(return_value=False)
    agent2 = MagicMock(spec=StoreAgent, agent_id="A2", name="Agent 2")
    agent2.execute_task = AsyncMock(side_effect=RuntimeError("Agent crashed!"))
    agent3 = MagicMock(spec=StoreAgent, agent_id="A3", name="Agent 3")
    agent3.execute_task = AsyncMock(return_value=True)
    # No need to call mock_isinstance_func anymore
    coordinator.register_agent(agent1)
    coordinator.register_agent(agent2)
    coordinator.register_agent(agent3)

    # Create and allocate tasks
    task1_id = coordinator.create_task(TaskType.PICKUP, "Task 1 Fail", 5, 2)
    task2_id = coordinator.create_task(TaskType.DELIVERY, "Task 2 Crash", 6, 3)
    task3_id = coordinator.create_task(TaskType.RESTOCKING, "Task 3 OK", 7, 4)
    coordinator.tasks[task1_id].status = TaskStatus.ALLOCATED
    coordinator.tasks[task1_id].assigned_agent_id = "A1"
    coordinator.tasks[task2_id].status = TaskStatus.ALLOCATED
    coordinator.tasks[task2_id].assigned_agent_id = "A2"
    coordinator.tasks[task3_id].status = TaskStatus.ALLOCATED
    coordinator.tasks[task3_id].assigned_agent_id = "A3"

    # Execute
    with caplog.at_level(logging.ERROR):
        await coordinator.execute_allocated_tasks()

    # Assertions
    agent1.execute_task.assert_awaited_once_with(coordinator.tasks[task1_id])
    agent2.execute_task.assert_awaited_once_with(coordinator.tasks[task2_id])
    agent3.execute_task.assert_awaited_once_with(coordinator.tasks[task3_id])

    # Check task status updated for exception case
    # Note: The agent's execute_task is responsible for status update on normal failure (return False)
    # The coordinator only handles exceptions during gather.
    assert coordinator.tasks[task1_id].status == TaskStatus.ALLOCATED # Status not changed by coordinator on False return
    assert coordinator.tasks[task2_id].status == TaskStatus.FAILED # Status changed by coordinator on Exception
    assert coordinator.tasks[task3_id].status == TaskStatus.ALLOCATED # Status not changed by coordinator on True return

    # Check error log for the exception case using caplog
    assert f"Error during execution of task {task2_id}" in caplog.text
    assert "Agent crashed!" in caplog.text

    # Check printed output for the error using capsys
    # captured = capsys.readouterr()
    # assert f"Error during execution of task {task2_id}" in captured.out
    # assert "Agent crashed!" in captured.out

@pytest.mark.asyncio
async def test_execute_allocated_tasks_agent_not_found(coordinator: RetailCoordinator, capsys):
    """Test handling when assigned agent is no longer registered."""
    # Create and allocate task
    task_id = coordinator.create_task(TaskType.PICKUP, "Task Agent Missing", 5, 2)
    task = coordinator.tasks[task_id]
    task.status = TaskStatus.ALLOCATED
    task.assigned_agent_id = "A_MISSING"
    # DO NOT register agent A_MISSING

    await coordinator.execute_allocated_tasks()

    # Check task status becomes FAILED
    assert task.status == TaskStatus.FAILED
    # Check warning logged
    captured = capsys.readouterr()
    assert f"Agent A_MISSING assigned to task {task_id} not found" in captured.out

# Placeholder tests
# async def test_execute_allocated_tasks...(): ... 