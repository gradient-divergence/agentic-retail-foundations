import logging
from datetime import datetime
from unittest.mock import patch

import pytest

# Module to test
from agents.bdi import InventoryBDIAgent
from models.inventory import InventoryItem, ProductInfo, SalesData

# --- Fixtures --- #


@pytest.fixture
def product_info() -> ProductInfo:
    return ProductInfo(
        product_id="P101",
        name="Test Coffee",
        category="Beverage",
        price=12.0,
        cost=6.0,
        supplier_id="SUP_A",
        lead_time_days=3,
        shelf_life_days=30,
        min_order_quantity=10,
    )


@pytest.fixture
def inventory_item() -> InventoryItem:
    return InventoryItem(
        product_id="P101",
        current_stock=50,
        reorder_point=20,
        optimal_stock=60,
        last_reorder_date=datetime(2024, 1, 1),
        pending_order_quantity=0,
        expected_delivery_date=None,
    )


@pytest.fixture
def sales_data() -> SalesData:
    # Avg = 5, Trend ~0
    return SalesData(product_id="P101", daily_sales=[4, 5, 6, 4, 5, 6, 5])


@pytest.fixture
def bdi_agent(product_info, inventory_item, sales_data) -> InventoryBDIAgent:
    agent = InventoryBDIAgent()
    # Pre-populate beliefs for tests
    agent.update_beliefs(
        new_products={product_info.product_id: product_info},
        new_inventory={inventory_item.product_id: inventory_item},
        new_sales={sales_data.product_id: sales_data},
        new_date=datetime(2024, 1, 15),
    )
    return agent


# --- Test Initialization & Beliefs --- #


def test_bdi_agent_initialization():
    """Test default initialization."""
    agent = InventoryBDIAgent()
    assert agent.products == {}
    assert agent.inventory == {}
    assert agent.sales_data == {}
    assert agent.active_intentions == []
    assert isinstance(agent.current_date, datetime)
    assert agent.goals["minimize_stockouts"] == 1.0  # Check one goal


def test_bdi_agent_update_beliefs(product_info, inventory_item, sales_data):
    """Test belief update mechanism."""
    agent = InventoryBDIAgent()
    test_date = datetime(2024, 2, 1)

    # Initial empty state
    assert agent.products == {}

    # Update all
    agent.update_beliefs(
        new_products={product_info.product_id: product_info},
        new_inventory={inventory_item.product_id: inventory_item},
        new_sales={sales_data.product_id: sales_data},
        new_date=test_date,
    )

    assert agent.products == {product_info.product_id: product_info}
    assert agent.inventory == {inventory_item.product_id: inventory_item}
    assert agent.sales_data == {sales_data.product_id: sales_data}
    assert agent.current_date == test_date
    # Check if update_beliefs correctly copied inventory to product info
    assert agent.products[product_info.product_id].inventory == inventory_item.current_stock
    # Check if current_price was set from price if missing (it wasn't missing here)
    assert agent.products[product_info.product_id].current_price == product_info.price

    # Test partial update
    agent.update_beliefs(new_date=datetime(2024, 3, 1))
    assert agent.current_date == datetime(2024, 3, 1)
    assert agent.inventory == {inventory_item.product_id: inventory_item}  # Should remain


# --- Test Observe --- #


@patch.object(InventoryBDIAgent, "_fetch_competitor_prices")
@patch.object(InventoryBDIAgent, "_fetch_supplier_lead_time")
def test_observe_success(
    mock_fetch_lead,
    mock_fetch_comp,
    bdi_agent: InventoryBDIAgent,
    product_info,
    inventory_item,
    sales_data,
):
    """Test the observe phase gathers correct data."""
    product_id = product_info.product_id
    mock_comp_prices = {"CompX": 11.50, "CompY": 12.50}
    mock_lead_time = 4  # Different from product_info default

    mock_fetch_comp.return_value = mock_comp_prices
    mock_fetch_lead.return_value = mock_lead_time

    observation = bdi_agent.observe(product_id)

    mock_fetch_comp.assert_called_once_with(product_id)
    mock_fetch_lead.assert_called_once_with(product_info.supplier_id)

    assert observation["current_price"] == product_info.current_price
    assert observation["cost"] == product_info.cost
    assert observation["inventory"] == inventory_item.current_stock
    assert observation["competitor_prices"] == mock_comp_prices
    assert observation["sales_last_7_days"] == sales_data.daily_sales[-7:]
    assert observation["lead_time"] == mock_lead_time
    assert observation["reorder_point"] == inventory_item.reorder_point
    assert observation["optimal_stock"] == inventory_item.optimal_stock

    # Verify product object was updated by observe
    assert bdi_agent.products[product_id].competitor_prices == mock_comp_prices
    assert bdi_agent.products[product_id].inventory == inventory_item.current_stock
    assert bdi_agent.products[product_id].sales_last_7_days == sales_data.daily_sales[-7:]


def test_observe_missing_data(bdi_agent: InventoryBDIAgent, caplog):
    """Test observe when product/inventory data is missing."""
    product_id_missing = "P_MISSING"

    with caplog.at_level(logging.WARNING):
        observation = bdi_agent.observe(product_id_missing)

    assert observation == {}
    assert f"Observe: missing product or inventory data for {product_id_missing}" in caplog.text


# --- Test Orient --- #


@pytest.mark.parametrize(
    "test_id, inventory_level, sales_7d, avg_comp_price, lead_time, expected_orientation",
    [
        # Case 1: Optimal Stock, Normal Sales
        (
            "optimal_normal",
            50,
            [4, 5, 6, 4, 5, 6, 5],
            12.0,
            3,
            {
                "price_position": "competitive",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "balanced",
                "days_of_supply": pytest.approx(50 / 5.0),
                "projected_daily_sales": 5.0,
            },
        ),
        # Case 2: Low Stock, High Sales -> Risk of Stockout
        (
            "low_stock_high_sales",
            15,
            [8, 9, 10, 8, 9, 10, 9],
            12.0,
            3,  # Inv < RP=20, DoS=15/9=1.6 < LT+3=6
            {
                "price_position": "competitive",
                "inventory_status": "low",
                "sales_assessment": "risk_of_stockout",
                "market_situation": "high_demand_low_supply",
                "days_of_supply": pytest.approx(15 / 9.0),
                "projected_daily_sales": 9.0,
            },
        ),
        # Case 3: High Stock, Low Sales -> Slow Moving
        (
            "high_stock_low_sales",
            100,
            [0, 1, 0, 0, 1, 0, 0],
            12.0,
            3,  # Inv > OS*1.5=90, Sales < RP*0.1=2
            {
                "price_position": "competitive",
                "inventory_status": "high",
                "sales_assessment": "slow_moving",
                "market_situation": "low_demand_high_supply",
                "days_of_supply": pytest.approx(100 / 0.29, 0.1),
                "projected_daily_sales": pytest.approx(0.29, 0.1),
            },
        ),
        # Case 4: Premium Price, Slow Sales -> Price Sensitive
        (
            "premium_slow",
            50,
            [0, 1, 0, 0, 1, 0, 0],
            10.0,
            3,  # Price=12 > 10*1.1
            {
                "price_position": "premium",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "balanced",
                "days_of_supply": pytest.approx(50 / 0.29, 0.1),
                "projected_daily_sales": pytest.approx(0.29, 0.1),
            },
        ),
        # Case 5: Discount Price, Normal Sales -> Underpriced
        (
            "discount_normal",
            50,
            [4, 5, 6, 4, 5, 6, 5],
            14.0,
            3,  # Price=12 < 14*0.9
            {
                "price_position": "discount",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "underpriced",
                "days_of_supply": pytest.approx(50 / 5.0),
                "projected_daily_sales": 5.0,
            },
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",  # Use test_id
)
def test_orient(
    bdi_agent: InventoryBDIAgent,
    product_info: ProductInfo,
    inventory_item: InventoryItem,
    test_id: str,
    inventory_level: int,
    sales_7d: list[int],
    avg_comp_price: float,
    lead_time: int,
    expected_orientation: dict,
):
    """Test the orient phase correctly analyzes various observations."""
    product_id = product_info.product_id

    # Create the observation dictionary for this scenario
    observation = {
        "inventory": inventory_level,
        "sales_last_7_days": sales_7d,
        "competitor_prices": {
            "CompX": avg_comp_price - 0.5,
            "CompY": avg_comp_price + 0.5,
        },
        "lead_time": lead_time,
        # Include other keys needed by orient based on observe output
        "current_price": product_info.current_price,  # Use fixture product's price
        "cost": product_info.cost,
        "reorder_point": inventory_item.reorder_point,
        "optimal_stock": inventory_item.optimal_stock,
    }

    # Need to update sales_data for trend calculation
    bdi_agent.sales_data[product_id] = SalesData(product_id=product_id, daily_sales=sales_7d * 2)  # Provide more history
    bdi_agent.inventory[product_id].current_stock = inventory_level  # Update inventory belief

    orientation = bdi_agent.orient(product_id, observation)

    # Assert calculated fields match expected
    for key, expected_value in expected_orientation.items():
        if isinstance(expected_value, float):
            assert orientation.get(key) == pytest.approx(expected_value), f"Mismatch on key: {key}"
        else:
            assert orientation.get(key) == expected_value, f"Mismatch on key: {key}"


def test_orient_missing_data(bdi_agent: InventoryBDIAgent, product_info, caplog):
    """Test orient returns empty dict if observation is missing."""
    product_id = product_info.product_id
    # Ensure product exists, but pass empty observation
    bdi_agent.update_beliefs(new_products={product_id: product_info})

    with caplog.at_level(logging.WARNING):
        orientation_empty = bdi_agent.orient(product_id, {})
        orientation_none = bdi_agent.orient(product_id, None)  # type: ignore

    assert orientation_empty == {}
    assert orientation_none == {}
    assert f"Orient: missing observation or product data for {product_id}" in caplog.text


# --- Test Deliberate --- #


@pytest.mark.parametrize(
    "test_id, mock_utilities, expected_goal_order",
    [
        # Case 1: Stockout prevention is highest priority
        (
            "stockout_high",
            {"stockout": 0.8, "excess": 0.1, "profit": 0.1, "fresh": 0.2},
            [
                "minimize_stockouts",
                "ensure_fresh_products",
                "minimize_excess_inventory",
                "maximize_profit_margin",
            ],
            # Utilities * weights: stockout=0.8*1=0.8, excess=0.1*0.7=0.07, profit=0.1*0.5=0.05, fresh=0.2*0.8=0.16
            # Order: stockout, fresh, excess, profit
        ),
        # Case 2: Excess inventory reduction is highest
        (
            "excess_high",
            {"stockout": 0.1, "excess": 0.9, "profit": 0.1, "fresh": 0.1},
            [
                "minimize_excess_inventory",
                "minimize_stockouts",
                "ensure_fresh_products",
                "maximize_profit_margin",
            ],
            # Utilities * weights: stockout=0.1, excess=0.63, profit=0.05, fresh=0.08
            # Order: excess, stockout, fresh, profit
        ),
        # Case 3: Profit is somehow highest (utility must be very high)
        (
            "profit_high",
            {"stockout": 0.1, "excess": 0.1, "profit": 0.9, "fresh": 0.1},
            [
                "maximize_profit_margin",
                "minimize_stockouts",
                "ensure_fresh_products",
                "minimize_excess_inventory",
            ],
            # Utilities * weights: stockout=0.1, excess=0.07, profit=0.45, fresh=0.08
            # Order: profit, stockout, fresh, excess
        ),
        # Case 4: Freshness is highest
        (
            "fresh_high",
            {"stockout": 0.1, "excess": 0.1, "profit": 0.1, "fresh": 0.9},
            [
                "ensure_fresh_products",
                "minimize_stockouts",
                "minimize_excess_inventory",
                "maximize_profit_margin",
            ],
            # Utilities * weights: stockout=0.1, excess=0.07, profit=0.05, fresh=0.72
            # Order: fresh, stockout, excess, profit
        ),
        # Case 5: All utilities low/zero
        (
            "all_low",
            {"stockout": 0.0, "excess": 0.0, "profit": 0.0, "fresh": 0.0},
            [],  # No goals above threshold
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",  # Use test_id
)
@patch.object(InventoryBDIAgent, "_evaluate_stockout_prevention")
@patch.object(InventoryBDIAgent, "_evaluate_excess_reduction")
@patch.object(InventoryBDIAgent, "_evaluate_profit_maximization")
@patch.object(InventoryBDIAgent, "_evaluate_freshness")
def test_deliberate(
    mock_eval_fresh,
    mock_eval_profit,
    mock_eval_excess,
    mock_eval_stockout,
    bdi_agent: InventoryBDIAgent,
    test_id: str,
    mock_utilities: dict,
    expected_goal_order: list[str],
):
    """Test the deliberate phase correctly prioritizes goals based on utility."""
    # Assign mock return values
    mock_eval_stockout.return_value = mock_utilities["stockout"]
    mock_eval_excess.return_value = mock_utilities["excess"]
    mock_eval_profit.return_value = mock_utilities["profit"]
    mock_eval_fresh.return_value = mock_utilities["fresh"]

    # Execute deliberate
    prioritized_goals = bdi_agent.deliberate()

    # Assert evaluation functions were called
    mock_eval_stockout.assert_called_once()
    mock_eval_excess.assert_called_once()
    mock_eval_profit.assert_called_once()
    mock_eval_fresh.assert_called_once()

    # Assert the returned goal list matches the expected order
    assert prioritized_goals == expected_goal_order


# --- Test Generate Intentions --- #


@patch.object(InventoryBDIAgent, "_plan_reorders")
@patch.object(InventoryBDIAgent, "_plan_inventory_reduction")
@patch.object(InventoryBDIAgent, "_plan_margin_optimization")
@patch.object(InventoryBDIAgent, "_plan_freshness_management")
def test_generate_intentions_calls_correct_planners(
    mock_plan_fresh,
    mock_plan_margin,
    mock_plan_excess,
    mock_plan_reorder,
    bdi_agent: InventoryBDIAgent,
):
    """Test generate_intentions calls the correct planning methods based on goals."""
    # Add a dummy intention to check it gets cleared
    bdi_agent.active_intentions = [{"action": "dummy"}]

    goals_order_1 = ["minimize_stockouts", "minimize_excess_inventory"]
    goals_order_2 = ["ensure_fresh_products", "maximize_profit_margin"]
    goals_order_3 = ["minimize_excess_inventory", "ensure_fresh_products"]
    goals_order_4 = []  # No goals

    # --- Test Case 1: Reorder + Excess --- #
    mock_plan_reorder.reset_mock()
    mock_plan_excess.reset_mock()
    mock_plan_margin.reset_mock()
    mock_plan_fresh.reset_mock()
    bdi_agent.generate_intentions(goals_order_1)
    mock_plan_reorder.assert_called_once()
    mock_plan_excess.assert_called_once()
    mock_plan_margin.assert_not_called()
    mock_plan_fresh.assert_not_called()
    # Check processed_products set is passed (by checking call args)
    assert isinstance(mock_plan_reorder.call_args.args[0], set)
    assert isinstance(mock_plan_excess.call_args.args[0], set)
    assert bdi_agent.active_intentions == []  # Should be cleared after execution normally, but check cleared at start

    # --- Test Case 2: Fresh + Margin --- #
    bdi_agent.active_intentions = [{"action": "dummy"}]  # Reset dummy
    mock_plan_reorder.reset_mock()
    mock_plan_excess.reset_mock()
    mock_plan_margin.reset_mock()
    mock_plan_fresh.reset_mock()
    bdi_agent.generate_intentions(goals_order_2)
    mock_plan_reorder.assert_not_called()
    mock_plan_excess.assert_not_called()
    mock_plan_margin.assert_called_once()
    mock_plan_fresh.assert_called_once()
    assert isinstance(mock_plan_fresh.call_args.args[0], set)
    assert isinstance(mock_plan_margin.call_args.args[0], set)

    # --- Test Case 3: Excess + Fresh --- #
    bdi_agent.active_intentions = [{"action": "dummy"}]  # Reset dummy
    mock_plan_reorder.reset_mock()
    mock_plan_excess.reset_mock()
    mock_plan_margin.reset_mock()
    mock_plan_fresh.reset_mock()
    bdi_agent.generate_intentions(goals_order_3)
    mock_plan_reorder.assert_not_called()
    mock_plan_excess.assert_called_once()
    mock_plan_margin.assert_not_called()
    mock_plan_fresh.assert_called_once()
    assert isinstance(mock_plan_excess.call_args.args[0], set)
    assert isinstance(mock_plan_fresh.call_args.args[0], set)

    # --- Test Case 4: No Goals --- #
    bdi_agent.active_intentions = [{"action": "dummy"}]  # Reset dummy
    mock_plan_reorder.reset_mock()
    mock_plan_excess.reset_mock()
    mock_plan_margin.reset_mock()
    mock_plan_fresh.reset_mock()
    bdi_agent.generate_intentions(goals_order_4)
    mock_plan_reorder.assert_not_called()
    mock_plan_excess.assert_not_called()
    mock_plan_margin.assert_not_called()
    mock_plan_fresh.assert_not_called()


# --- Test Execute Intentions --- #


@patch.object(InventoryBDIAgent, "_execute_reorder", return_value=True)  # Mock success
@patch.object(InventoryBDIAgent, "_execute_discount", return_value=True)
@patch.object(InventoryBDIAgent, "_execute_promotion", return_value=True)
@patch.object(InventoryBDIAgent, "_execute_perishable_discount", return_value=False)  # Mock failure
def test_execute_intentions_prioritization_and_filtering(
    mock_exec_perish,
    mock_exec_promo,
    mock_exec_discount,
    mock_exec_reorder,
    bdi_agent: InventoryBDIAgent,
):
    """Test execute_intentions sorts by priority and processes one per product."""
    # Setup intentions with different priorities and products
    intention_reorder_p1_high = {
        "action": "reorder",
        "product_id": "P1",
        "priority": 0.9,
    }
    intention_discount_p1_low = {
        "action": "discount",
        "product_id": "P1",
        "priority": 0.3,
    }
    intention_promote_p2 = {"action": "promote", "product_id": "P2", "priority": 0.7}
    intention_perish_p3 = {
        "action": "discount_perishable",
        "product_id": "P3",
        "priority": 0.5,
    }
    intention_unknown_p4 = {"action": "unknown", "product_id": "P4", "priority": 0.8}
    intention_no_pid = {"action": "reorder", "priority": 1.0}  # Should be skipped

    # Add in non-priority order
    bdi_agent.active_intentions = [
        intention_discount_p1_low,
        intention_promote_p2,
        intention_perish_p3,
        intention_no_pid,
        intention_reorder_p1_high,  # Highest priority for P1
        intention_unknown_p4,
    ]
    initial_intentions = bdi_agent.active_intentions.copy()

    # Execute
    executed_actions = bdi_agent.execute_intentions()

    # --- Assertions --- #
    # 1. Check which execute methods were called
    #    - Should call reorder for P1 (highest priority)
    #    - Should call promote for P2
    #    - Should call perishable_discount for P3
    #    - Should NOT call discount for P1 (lower priority)
    #    - Should NOT call anything for P4 (unknown action)
    #    - Should NOT call anything for no_pid intention
    mock_exec_reorder.assert_called_once_with(intention_reorder_p1_high)
    mock_exec_discount.assert_not_called()
    mock_exec_promo.assert_called_once_with(intention_promote_p2)
    mock_exec_perish.assert_called_once_with(intention_perish_p3)

    # 2. Check returned executed actions list
    #    - Should include intentions where mock returned True
    #    - Should NOT include perish_p3 (mock returned False)
    #    - Should NOT include unknown_p4 or no_pid
    assert len(executed_actions) == 2
    assert intention_reorder_p1_high in executed_actions
    assert intention_promote_p2 in executed_actions
    assert intention_perish_p3 not in executed_actions
    assert intention_discount_p1_low not in executed_actions
    assert intention_unknown_p4 not in executed_actions
    assert intention_no_pid not in executed_actions

    # 3. Check active_intentions list is cleared
    assert bdi_agent.active_intentions == []


@patch.object(InventoryBDIAgent, "_execute_reorder", side_effect=Exception("Execution Error!"))
def test_execute_intentions_handles_execution_error(mock_exec_reorder, bdi_agent: InventoryBDIAgent, caplog):
    """Test that execute_intentions handles errors during execution of one intention."""
    intention_reorder = {"action": "reorder", "product_id": "P1", "priority": 0.9}
    intention_promote = {"action": "promote", "product_id": "P2", "priority": 0.7}
    bdi_agent.active_intentions = [intention_reorder, intention_promote]

    # Mock other execute methods to avoid interference and capture logs
    with (
        patch.object(InventoryBDIAgent, "_execute_promotion", return_value=True) as mock_exec_promo,
        caplog.at_level(logging.ERROR),
    ):
        executed_actions = bdi_agent.execute_intentions()

    # Reorder should have failed and logged error
    mock_exec_reorder.assert_called_once_with(intention_reorder)
    assert "Error executing intention" in caplog.text
    assert "Execution Error!" in caplog.text

    # Promote should still have been attempted and succeeded
    mock_exec_promo.assert_called_once_with(intention_promote)

    # Only the successful promotion should be returned
    assert executed_actions == [intention_promote]

    # Intentions list should still be cleared
    assert bdi_agent.active_intentions == []


# Placeholder tests for evaluation methods
# def test_evaluate_...(): ...

# Placeholder tests for planning methods
# def test_plan_...(): ...

# Placeholder tests for execution methods (specific logic)
# def test_execute_...(): ...

# Placeholder tests for run_cycle
# def test_run_cycle(): ...

# Placeholder tests for helpers
# def test_helpers(): ...
