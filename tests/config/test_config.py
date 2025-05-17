from config.config import DynamicPricingMDPConfig, QLearningAgentConfig


def test_dynamic_pricing_mdp_config_defaults():
    """Test DynamicPricingMDPConfig initializes with correct default values."""
    config = DynamicPricingMDPConfig()
    assert config.initial_inventory == 100
    assert config.season_length_weeks == 12
    assert config.base_price == 50.0
    assert config.base_demand == 10.0
    assert config.price_elasticity == 1.5
    assert config.holding_cost_per_unit == 0.5
    assert config.end_season_salvage_value == 15.0
    assert config.available_discounts == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_dynamic_pricing_mdp_config_custom():
    """Test DynamicPricingMDPConfig initialization with custom values."""
    custom_inventory = 50
    custom_discounts = [0.0, 0.25, 0.5]
    config = DynamicPricingMDPConfig(initial_inventory=custom_inventory, available_discounts=custom_discounts)
    assert config.initial_inventory == custom_inventory
    assert config.available_discounts == custom_discounts
    # Check a default value is still correct
    assert config.base_price == 50.0


def test_dynamic_pricing_mdp_config_default_factory():
    """Test that the default_factory creates separate list instances."""
    config1 = DynamicPricingMDPConfig()
    config2 = DynamicPricingMDPConfig()
    assert config1.available_discounts == config2.available_discounts
    assert config1.available_discounts is not config2.available_discounts
    # Modify one list, check the other is unaffected
    config1.available_discounts.append(0.6)
    assert config2.available_discounts == [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def test_qlearning_agent_config_defaults():
    """Test QLearningAgentConfig initializes with correct default values."""
    config = QLearningAgentConfig()
    assert config.learning_rate == 0.1
    assert config.discount_factor == 0.95
    assert config.exploration_rate == 1.0
    assert config.min_exploration_rate == 0.01
    assert config.exploration_decay == 0.995
    assert config.action_space_size == 6


def test_qlearning_agent_config_custom():
    """Test QLearningAgentConfig initialization with custom values."""
    custom_lr = 0.05
    custom_action_space = 10
    config = QLearningAgentConfig(learning_rate=custom_lr, action_space_size=custom_action_space)
    assert config.learning_rate == custom_lr
    assert config.action_space_size == custom_action_space
    # Check a default value
    assert config.discount_factor == 0.95
