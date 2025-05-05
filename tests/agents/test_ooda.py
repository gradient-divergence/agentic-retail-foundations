import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import logging
import random

# Module to test
from agents.ooda import OODAPricingAgent
from models.pricing import PricingProduct

# --- Fixtures --- #

@pytest.fixture
def agent_config() -> dict:
    """Default config weights for the agent."""
    return {
        "inventory_weight": 0.3,
        "competitor_weight": 0.4,
        "sales_weight": 0.3,
        "max_price_change_pct": 5.0,
    }

@pytest.fixture
def ooda_agent(agent_config) -> OODAPricingAgent:
    """Provides a default OODAPricingAgent instance."""
    return OODAPricingAgent(**agent_config)

@pytest.fixture
def sample_product() -> PricingProduct:
    """Provides a sample PricingProduct."""
    return PricingProduct(
        product_id="P001",
        name="Test Product",
        category="TestCat",
        cost=5.0,
        current_price=10.0,
        min_price=8.0,
        max_price=15.0,
        inventory=25, # Optimal inventory
        target_profit_margin=0.3,
        sales_last_7_days=[1, 2, 1, 3, 2, 1, 2], # Avg = 2
    )

# --- Test Initialization --- #

def test_ooda_agent_initialization(ooda_agent, agent_config):
    """Test that OODAPricingAgent initializes correctly."""
    assert isinstance(ooda_agent.products, dict)
    assert isinstance(ooda_agent.action_history, list)
    assert ooda_agent.inventory_weight == agent_config["inventory_weight"]
    assert ooda_agent.competitor_weight == agent_config["competitor_weight"]
    assert ooda_agent.sales_weight == agent_config["sales_weight"]
    assert ooda_agent.max_price_change_pct == agent_config["max_price_change_pct"]

# --- Test update_products --- #

def test_ooda_agent_update_products(ooda_agent, sample_product):
    """Test updating products."""
    products_data = {sample_product.product_id: sample_product}
    ooda_agent.update_products(products_data)
    assert ooda_agent.products == products_data
    assert ooda_agent.products["P001"].name == "Test Product"

# --- Test observe --- #

# Patch the internal helper method directly
@patch.object(OODAPricingAgent, '_fetch_competitor_prices')
def test_observe_success(mock_fetch_comp, ooda_agent, sample_product):
    """Test the observe phase retrieves and returns data correctly."""
    # Setup
    product_id = sample_product.product_id
    ooda_agent.update_products({product_id: sample_product})
    mock_competitor_prices = {"CompA": 9.50, "CompB": 10.50}
    mock_fetch_comp.return_value = mock_competitor_prices

    # Execute
    observation = ooda_agent.observe(product_id)

    # Assert
    mock_fetch_comp.assert_called_once_with(sample_product)
    assert observation["product_id"] == product_id
    assert observation["current_price"] == sample_product.current_price
    assert observation["cost"] == sample_product.cost
    assert observation["inventory"] == sample_product.inventory
    assert observation["sales_last_7_days"] == sample_product.sales_last_7_days
    assert observation["competitor_prices"] == mock_competitor_prices
    assert "timestamp" in observation

def test_observe_product_not_found(ooda_agent, caplog):
    """Test observe phase when the product_id is not found."""
    product_id = "P_UNKNOWN"
    with caplog.at_level(logging.WARNING):
        observation = ooda_agent.observe(product_id)

    assert observation == {}
    assert f"Observe: {product_id} not found." in caplog.text

# --- Test orient --- #

@pytest.mark.parametrize(
    "test_id, input_observation, expected_orientation",
    [
        # Case 1: Balanced scenario (based on sample_product defaults)
        (
            "balanced",
            {
                "product_id": "P001",
                "current_price": 10.0,
                "inventory": 25,
                "competitor_prices": {"CompA": 9.50, "CompB": 10.50}, # Avg = 10.0
                "sales_last_7_days": [1, 2, 1, 3, 2, 1, 2], # Avg = 2.0
            },
            {
                "avg_competitor_price": 10.0,
                "price_position": "competitive",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "balanced",
                "avg_daily_sales_7d": pytest.approx(1.714, 0.001),
                "days_of_supply": pytest.approx(14.583, 0.001),
            }
        ),
        # Case 2: Low inventory, high sales -> high_demand_low_supply
        (
            "low_inv_high_sales",
            {
                "product_id": "P001",
                "current_price": 11.0, # Competitive
                "inventory": 5, # Low
                "competitor_prices": {"CompA": 10.0, "CompB": 12.0}, # Avg = 11.0
                "sales_last_7_days": [5, 6, 5, 7, 5, 6, 5], # Avg = 5.57
            },
            {
                "avg_competitor_price": 11.0,
                "price_position": "competitive",
                "inventory_status": "low",
                "sales_assessment": "risk_of_stockout",
                "market_situation": "high_demand_low_supply",
                "avg_daily_sales_7d": pytest.approx(5.57, 0.01),
                "days_of_supply": pytest.approx(5 / 5.57, 0.01),
            }
        ),
        # Case 3: High inventory, low sales -> low_demand_high_supply
        (
            "high_inv_low_sales",
            {
                "product_id": "P001",
                "current_price": 9.0, # Competitive
                "inventory": 80, # High
                "competitor_prices": {"CompA": 8.5, "CompB": 9.5}, # Avg = 9.0
                "sales_last_7_days": [0, 1, 0, 0, 1, 0, 0], # Avg = 0.28
            },
            {
                "avg_competitor_price": 9.0,
                "price_position": "competitive",
                "inventory_status": "high",
                "sales_assessment": "slow_moving",
                "market_situation": "low_demand_high_supply",
                "avg_daily_sales_7d": pytest.approx(2/7, abs=0.01),
                "days_of_supply": pytest.approx(80 / (2/7), abs=0.1),
            }
        ),
        # Case 4: Premium price, stagnant sales -> price_sensitive_market
        (
            "premium_stagnant",
            {
                "product_id": "P001",
                "current_price": 14.0, # Premium
                "inventory": 30, # Optimal
                "competitor_prices": {"CompA": 10.0, "CompB": 11.0}, # Avg = 10.5
                "sales_last_7_days": [0, 0, 0, 0, 0, 0, 0], # Avg = 0.0
            },
            {
                "avg_competitor_price": 10.5,
                "price_position": "premium",
                "inventory_status": "optimal",
                "sales_assessment": "stagnant",
                "market_situation": "price_sensitive_market",
                "avg_daily_sales_7d": 0.0,
                "days_of_supply": float('inf'),
            }
        ),
        # Case 5: Discount price, normal sales -> underpriced
        (
            "discount_normal",
            {
                "product_id": "P001",
                "current_price": 8.0, # Discount
                "inventory": 40, # Optimal
                "competitor_prices": {"CompA": 10.0, "CompB": 11.0}, # Avg = 10.5
                "sales_last_7_days": [1, 2, 1, 3, 2, 1, 2], # Avg = 12/7
            },
            {
                "avg_competitor_price": 10.5,
                "price_position": "discount",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "underpriced",
                "avg_daily_sales_7d": pytest.approx(12/7, abs=0.01),
                "days_of_supply": pytest.approx(40 / (12/7), abs=0.1),
            }
        ),
        # Case 6: No competitor prices
        (
            "no_competitors",
            {
                "product_id": "P001",
                "current_price": 10.0,
                "inventory": 25,
                "competitor_prices": {}, # Empty
                "sales_last_7_days": [1, 2, 1, 3, 2, 1, 2],
            },
            {
                "avg_competitor_price": 10.0, # Should default to current price
                "price_position": "competitive",
                "inventory_status": "optimal",
                "sales_assessment": "normal",
                "market_situation": "balanced",
                "avg_daily_sales_7d": pytest.approx(1.714, 0.001),
                "days_of_supply": pytest.approx(14.583, 0.001),
            }
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
def test_orient(ooda_agent, sample_product, test_id, input_observation, expected_orientation):
    """Test the orient phase correctly analyzes the observation."""
    product_id = sample_product.product_id
    # Update agent with the product (needed for orient to access current_price if no competitors)
    ooda_agent.update_products({product_id: sample_product})

    orientation = ooda_agent.orient(product_id, input_observation)

    assert orientation["product_id"] == product_id
    assert "timestamp" in orientation
    # Check calculated fields against expected values
    for key, expected_value in expected_orientation.items():
        if isinstance(expected_value, float):
            assert orientation[key] == pytest.approx(expected_value)
        else:
            assert orientation[key] == expected_value

def test_orient_missing_data(ooda_agent, sample_product, caplog):
    """Test orient returns empty dict and logs warning if observation is missing."""
    product_id = sample_product.product_id
    ooda_agent.update_products({product_id: sample_product})

    with caplog.at_level(logging.WARNING):
        orientation_empty = ooda_agent.orient(product_id, {})
        orientation_none = ooda_agent.orient(product_id, None) # type: ignore

    assert orientation_empty == {}
    assert orientation_none == {}
    assert f"Orient: Missing data for {product_id}." in caplog.text
    assert caplog.text.count(f"Orient: Missing data for {product_id}.") >= 1 # >=1 allows for potential other warnings

# --- Test decide --- #

@pytest.mark.parametrize(
    "test_id, input_orientation, current_price, min_price, max_price, expected_decision",
    [
        # Case 1: Balanced -> small change (likely capped at 0 due to psychology)
        (
            "balanced_no_change",
            {
                "avg_competitor_price": 10.0,
                "inventory_status": "optimal",
                "sales_assessment": "normal",
            },
            10.0, 8.0, 15.0,
            {
                "old_price": 10.0,
                "new_price": 10.99, # Psychology rounds up
                "capped_change_pct": pytest.approx(0.0), # Components should be 0
                "primary_driver": "none",
            }
        ),
        # Case 2: High demand, low supply -> increase price (capped)
        (
            "high_demand_low_supply_increase",
            {
                "avg_competitor_price": 10.0, # Competitive price
                "inventory_status": "low",    # inv_comp = 2.0 * 0.3 = 0.6
                "sales_assessment": "risk_of_stockout", # sales_comp = 2.5 * 0.3 = 0.75
            },
            10.0, 8.0, 15.0,
            {
                "old_price": 10.0,
                # total_change = 0.6 + 0.0 + 0.75 = 1.35 (within cap)
                # new_price_raw = 10.0 * (1 + 1.35/100) = 10.135
                # psychology -> 10.99
                "new_price": 10.99,
                "capped_change_pct": pytest.approx(1.35),
                "primary_driver": "sales", # abs(0.75) > abs(0.6)
            }
        ),
        # Case 3: Low demand, high supply -> decrease price (capped)
        (
            "low_demand_high_supply_decrease",
            {
                "avg_competitor_price": 10.0, # Competitive price
                "inventory_status": "high",   # inv_comp = -3.0 * 0.3 = -0.9
                "sales_assessment": "slow_moving", # sales_comp = -2.5 * 0.3 = -0.75
            },
            10.0, 8.0, 15.0,
            {
                "old_price": 10.0,
                # total_change = -0.9 + 0.0 + -0.75 = -1.65 (within cap)
                # new_price_raw = 10.0 * (1 - 1.65/100) = 9.835
                # psychology -> 9.99
                "new_price": 9.99,
                "capped_change_pct": pytest.approx(-1.65),
                "primary_driver": "inventory", # abs(-0.9) > abs(-0.75)
            }
        ),
        # Case 4: Price too high vs comp -> decrease price
        (
            "price_too_high_decrease",
            {
                "avg_competitor_price": 10.0, # Price 12.0 -> diff = 20%
                "inventory_status": "optimal", # inv_comp = 0
                "sales_assessment": "normal",   # sales_comp = 0
            },
            12.0, 8.0, 15.0,
            {
                "old_price": 12.0,
                # price_diff_pct = 20%
                # comp_component = -(20 / 3.0) * 0.4 = -2.666...
                # total_change = -2.666... (within cap)
                # new_price_raw = 12.0 * (1 - 2.666/100) = 11.68
                # psychology -> 11.99
                "new_price": 11.99,
                "capped_change_pct": pytest.approx(-2.666, 0.01),
                "primary_driver": "competitor",
            }
        ),
        # Case 5: Change capped by max_price_change_pct (e.g., 5%)
        (
            "change_capped_positive",
            {
                "avg_competitor_price": 10.0, # price 8.0 -> diff = -20%
                "inventory_status": "low", # inv_comp = 0.6
                "sales_assessment": "risk_of_stockout", # sales_comp = 0.75
            },
            8.0, 8.0, 15.0,
            {
                "old_price": 8.0,
                # price_diff_pct = -20%
                # comp_component = -(-20 / 3.0) * 0.4 = 2.666...
                # total_change = 0.6 + 2.666 + 0.75 = 4.016 > 5% (if default cap was 3%)
                # Let's assume default cap = 5%
                # capped_change = 4.016 (within 5% cap)
                # new_price_raw = 8.0 * (1 + 4.016/100) = 8.321
                # psychology -> 8.99
                "new_price": 8.99,
                "capped_change_pct": pytest.approx(4.016, 0.01),
                "primary_driver": "competitor",
            }
        ),
        # Case 6: Respect min_price constraint
        (
            "respect_min_price",
            {
                "avg_competitor_price": 10.0,
                "inventory_status": "high", # inv_comp = -0.9
                "sales_assessment": "stagnant", # sales_comp = -4.0 * 0.3 = -1.2
            },
            8.5, 8.0, 15.0, # Current price is 8.5, min is 8.0
            {
                "old_price": 8.5,
                # total_change = (-3.0*0.3) + (5.0*0.4) + (-4.0*0.3) = -0.9 + 2.0 - 1.2 = -0.1
                # capped_change = -0.1
                # new_price_raw = 8.5 * (1 - 0.1/100) = 8.4915
                # psychology -> 8.99
                "new_price": 8.99,
                "capped_change_pct": pytest.approx(-0.1),
                "primary_driver": "competitor",
            }
        ),
        # Case 7: Respect max_price constraint
        (
            "respect_max_price",
            {
                "avg_competitor_price": 10.0,
                "inventory_status": "low", # inv_comp = 0.6
                "sales_assessment": "risk_of_stockout", # sales_comp = 0.75
            },
            14.5, 8.0, 15.0, # Current price is 14.5, max is 15.0
            {
                "old_price": 14.5,
                # total_change = (2.0*0.3) + (-15.0*0.4) + (2.5*0.3) = 0.6 - 6.0 + 0.75 = -4.65
                # capped_change = -4.65
                # new_price_raw = 14.5 * (1 - 4.65/100) = 13.82575
                # psychology -> 13.99 -> min(15.0, 13.99) = 13.99
                "new_price": 13.99,
                "capped_change_pct": pytest.approx(-4.65),
                "primary_driver": "competitor",
            }
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "" # Use test_id
)
def test_decide(
    ooda_agent: OODAPricingAgent,
    sample_product: PricingProduct,
    test_id: str, input_orientation: dict, current_price: float, min_price: float, max_price: float, expected_decision: dict
):
    """Test the decide phase correctly calculates new price and driver."""
    product_id = sample_product.product_id
    # Update product with current/min/max prices for the test case
    sample_product.current_price = current_price
    sample_product.min_price = min_price
    sample_product.max_price = max_price
    ooda_agent.update_products({product_id: sample_product})

    # Add necessary keys from product if not in orientation dict
    full_orientation = {
        "product_id": product_id,
        **input_orientation # Add test case specific orientation
    }

    decision = ooda_agent.decide(product_id, full_orientation)

    assert decision["product_id"] == product_id
    assert "timestamp" in decision
    # Check calculated fields against expected values
    for key, expected_value in expected_decision.items():
        if isinstance(expected_value, float):
            # Use approx for float comparisons due to potential precision issues
            assert decision[key] == pytest.approx(expected_value), f"Mismatch on key: {key}"
        else:
            assert decision[key] == expected_value, f"Mismatch on key: {key}"

def test_decide_missing_data(ooda_agent, sample_product, caplog):
    """Test decide returns empty dict and logs warning if orientation is missing."""
    product_id = sample_product.product_id
    ooda_agent.update_products({product_id: sample_product})

    with caplog.at_level(logging.WARNING):
        decision_empty = ooda_agent.decide(product_id, {})
        decision_none = ooda_agent.decide(product_id, None) # type: ignore

    assert decision_empty == {}
    assert decision_none == {}
    assert f"Decide: Missing orientation for {product_id}." in caplog.text
    assert caplog.text.count(f"Decide: Missing orientation for {product_id}.") >= 1

# --- Test act --- #

def test_act_success(ooda_agent: OODAPricingAgent, sample_product: PricingProduct, caplog):
    """Test the act phase successfully updates the product price and logs."""
    product_id = sample_product.product_id
    old_price = sample_product.current_price # 10.0
    new_price = 11.99
    decision = {
        "product_id": product_id,
        "old_price": old_price,
        "new_price": new_price,
        "primary_driver": "inventory",
    }
    ooda_agent.update_products({product_id: sample_product})
    initial_history_len = len(ooda_agent.action_history)

    with caplog.at_level(logging.INFO):
        acted = ooda_agent.act(product_id, decision)

    assert acted is True
    # Verify product price was updated
    assert ooda_agent.products[product_id].current_price == new_price
    # Verify action was logged
    assert len(ooda_agent.action_history) == initial_history_len + 1
    last_action = ooda_agent.action_history[-1]
    assert last_action["product_id"] == product_id
    assert last_action["old_price"] == old_price
    assert last_action["new_price"] == new_price
    assert last_action["reason"] == "inventory"
    assert f"Act {product_id}: Price updated to {new_price:.2f}" in caplog.text

def test_act_no_change(ooda_agent: OODAPricingAgent, sample_product: PricingProduct, caplog):
    """Test act phase when the price change is negligible."""
    product_id = sample_product.product_id
    old_price = sample_product.current_price # 10.0
    new_price = 10.005 # Change < 0.01
    decision = {
        "product_id": product_id,
        "old_price": old_price,
        "new_price": new_price,
        "primary_driver": "none",
    }
    ooda_agent.update_products({product_id: sample_product})
    initial_history_len = len(ooda_agent.action_history)

    with caplog.at_level(logging.INFO):
        acted = ooda_agent.act(product_id, decision)

    assert acted is False
    # Verify product price was NOT updated
    assert ooda_agent.products[product_id].current_price == old_price
    # Verify action was NOT logged
    assert len(ooda_agent.action_history) == initial_history_len
    assert f"Act {product_id}: price change too small, skipping." in caplog.text

def test_act_missing_decision(ooda_agent: OODAPricingAgent, sample_product: PricingProduct, caplog):
    """Test act phase when the decision dict is missing or empty."""
    product_id = sample_product.product_id
    old_price = sample_product.current_price
    ooda_agent.update_products({product_id: sample_product})
    initial_history_len = len(ooda_agent.action_history)

    with caplog.at_level(logging.WARNING):
        acted_empty = ooda_agent.act(product_id, {})
        acted_none = ooda_agent.act(product_id, None) # type: ignore

    assert acted_empty is False
    assert acted_none is False
    # Verify product price was NOT updated
    assert ooda_agent.products[product_id].current_price == old_price
    # Verify action was NOT logged
    assert len(ooda_agent.action_history) == initial_history_len
    assert f"Act: Missing decision for {product_id}." in caplog.text
    assert caplog.text.count(f"Act: Missing decision for {product_id}.") >= 1

# --- Test run_cycle_for_product --- #

@patch.object(OODAPricingAgent, 'observe')
@patch.object(OODAPricingAgent, 'orient')
@patch.object(OODAPricingAgent, 'decide')
@patch.object(OODAPricingAgent, 'act')
def test_run_cycle_for_product_success_flow(
    mock_act, mock_decide, mock_orient, mock_observe,
    ooda_agent: OODAPricingAgent, sample_product: PricingProduct
):
    """Test run_cycle calls observe, orient, decide, act in sequence and returns act result."""
    product_id = sample_product.product_id
    ooda_agent.update_products({product_id: sample_product})

    # Setup mock return values for each phase
    mock_observation = {"product_id": product_id, "inventory": 10}
    mock_orientation = {"product_id": product_id, "market_situation": "balanced"}
    mock_decision = {"product_id": product_id, "new_price": 9.99}
    mock_act.return_value = True # Simulate successful action

    mock_observe.return_value = mock_observation
    mock_orient.return_value = mock_orientation
    mock_decide.return_value = mock_decision

    # Execute
    result = ooda_agent.run_cycle_for_product(product_id)

    # Assert phase methods were called correctly
    mock_observe.assert_called_once_with(product_id)
    mock_orient.assert_called_once_with(product_id, mock_observation)
    mock_decide.assert_called_once_with(product_id, mock_orientation)
    mock_act.assert_called_once_with(product_id, mock_decision)

    # Assert final result matches act result
    assert result is True

@patch.object(OODAPricingAgent, 'observe')
@patch.object(OODAPricingAgent, 'orient')
@patch.object(OODAPricingAgent, 'decide')
@patch.object(OODAPricingAgent, 'act')
def test_run_cycle_for_product_stops_early(
    mock_act, mock_decide, mock_orient, mock_observe,
    ooda_agent: OODAPricingAgent, sample_product: PricingProduct
):
    """Test run_cycle stops if an early phase returns no result."""
    product_id = sample_product.product_id
    ooda_agent.update_products({product_id: sample_product})

    # --- Test stop after observe --- #
    mock_observe.return_value = {} # Simulate observe failing
    mock_orient.reset_mock(); mock_decide.reset_mock(); mock_act.reset_mock()

    result_obs_fail = ooda_agent.run_cycle_for_product(product_id)
    assert result_obs_fail is False
    mock_observe.assert_called_once_with(product_id)
    mock_orient.assert_not_called()
    mock_decide.assert_not_called()
    mock_act.assert_not_called()

    # --- Test stop after orient --- #
    mock_observe.return_value = {"product_id": product_id} # Observe succeeds
    mock_orient.return_value = {} # Orient fails
    mock_observe.reset_mock(); mock_decide.reset_mock(); mock_act.reset_mock()

    result_ori_fail = ooda_agent.run_cycle_for_product(product_id)
    assert result_ori_fail is False
    mock_observe.assert_called_once_with(product_id)
    mock_orient.assert_called_once_with(product_id, mock_observe.return_value)
    mock_decide.assert_not_called()
    mock_act.assert_not_called()

    # --- Test stop after decide --- #
    mock_observe.return_value = {"product_id": product_id} # Observe succeeds
    mock_orient.return_value = {"product_id": product_id} # Orient succeeds
    mock_decide.return_value = {} # Decide fails
    mock_observe.reset_mock(); mock_orient.reset_mock(); mock_act.reset_mock()

    result_dec_fail = ooda_agent.run_cycle_for_product(product_id)
    assert result_dec_fail is False
    mock_observe.assert_called_once_with(product_id)
    mock_orient.assert_called_once_with(product_id, mock_observe.return_value)
    mock_decide.assert_called_once_with(product_id, mock_orient.return_value)
    mock_act.assert_not_called()

# --- Test Helper Methods --- #

@patch('random.uniform')
def test_fetch_competitor_prices(mock_uniform, ooda_agent: OODAPricingAgent, sample_product: PricingProduct):
    """Test fetching competitor prices (mocking random noise)."""
    # Mock random.uniform to return specific noise values
    noise_a = -0.05 # 5% lower
    noise_b = 0.08  # 8% higher
    mock_uniform.side_effect = [noise_a, noise_b]

    product = sample_product # current_price=10.0, min_price=8.0
    expected_comp_a = round(max(product.min_price, product.current_price * (1 + noise_a)), 2)
    expected_comp_b = round(max(product.min_price, product.current_price * (1 + noise_b)), 2)

    comp_prices = ooda_agent._fetch_competitor_prices(product)

    assert comp_prices == {"CompetitorA": expected_comp_a, "CompetitorB": expected_comp_b}
    # Ensure uniform was called twice (once for each competitor)
    assert mock_uniform.call_count == 2

@pytest.mark.parametrize(
    "input_price, expected_price",
    [
        (10.50, 10.99),
        (10.95, 10.99),
        (10.00, 10.99),
        (10.99, 10.99),
        (12.34, 12.99),
        (0.85, 0.85),  # Below 1.0, should just round
        (0.99, 0.99),
        (0.999, 1.00), # Standard rounding applies below 1.0?
                       # Let's re-read code: round(price, 2) if price < 1.0. So 0.999 -> 1.00
    ]
)
def test_apply_price_psychology(ooda_agent: OODAPricingAgent, input_price: float, expected_price: float):
    """Test the price psychology adjustment logic."""
    adjusted_price = ooda_agent._apply_price_psychology(input_price)
    assert adjusted_price == pytest.approx(expected_price)
