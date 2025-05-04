# Core Concepts: Agents

<!--
Describe the different types of agent architectures implemented or planned
(e.g., BDI, OODA loop, reactive agents).
Explain the base agent classes and how they are structured.
Provide high-level examples of how agents interact.
-->

# Agents & Protocols

This section describes the different types of agents and coordination protocols implemented in the framework.

## Agent Roles

Agents in this framework represent distinct functional units or decision-making entities within a retail environment. They communicate and collaborate to achieve higher-level goals.

### Core Agent Types

These agents implement specific algorithms or handle core functionalities:

-   **Base Agent (`agents/base.py`):** See [API Reference](/reference/agents.md).
-   **LLM Agent (`agents/llm.py`):** See [API Reference](/reference/agents.md).
-   **Orchestrator (`agents/orchestrator.py`):** See [API Reference](/reference/agents.md).
-   **Q-Learning Agent (`agents/qlearning.py`):** Used for reinforcement learning tasks like dynamic pricing. See [Dynamic Pricing Demo](/examples/index.md#decision-making-frameworks).
-   **Bayesian Agent (`agents/bayesian.py`):** Used for probabilistic reasoning, e.g., recommendations. See [Recommendation Demo](/examples/index.md#decision-making-frameworks).
-   **Causal Analysis Agent (`agents/promotion_causal.py`):** Performs causal inference for promotion effectiveness.
-   **OODA/BDI Agents (`agents/ooda.py`, `agents/bdi.py`):** Alternative agent architectures.
-   **Fulfillment Agent (`agents/fulfillment.py`):** Manages order picking/packing flows.
-   **Inventory Orchestration Agent (`agents/inventory_orchestration.py`):** Event-driven inventory allocation.
-   **Dynamic Pricing Agent (`agents/dynamic_pricing_feedback.py`):** Real-time pricing adjustments. See [Feedback Loop Demo](/examples/index.md#decision-making-frameworks).
-   **(S,s) Inventory Agent (`agents/inventory.py`):** Implements the classic (s, S) inventory policy.
-   **Computer Vision Agent (`agents/cv.py`):** Placeholder for visual monitoring.
-   **Sensor Agent (`agents/sensor.py`):** Simulates sensor data processing.
-   **Store Agent (`agents/store.py`):** Represents a store in coordination protocols.

### Cross-Functional Agents

Located in `agents/cross_functional/`, these simulate specific departments:

-   `CustomerServiceAgent`
-   `MarketingAgent`
-   `PricingAgent`
-   `StoreOpsAgent`
-   `SupplyChainAgent`

## Coordination Protocols

Located in `agents/protocols/`, these implement interaction patterns:

-   **Contract Net (`contract_net.py`):** Task allocation via bidding. See [API Reference](/reference/agents.md) and [CNP Demo](/examples/index.md#agent-communication--protocols).
-   **Auction (`auction.py`):** Supplier selection via auctions. See [API Reference](/reference/agents.md) and [Auction Demo](/examples/index.md#agent-communication--protocols).
-   **Inventory Sharing (`inventory_sharing.py`):** Cooperative stock transfers. See [API Reference](/reference/agents.md) and [Sharing Demo](/examples/index.md#agent-communication--protocols).

## Specialized Coordinators

Located in `agents/coordinators/`:

-   **Product Launch (`product_launch.py`):** Orchestrates new product introductions. See [API Reference](/reference/agents.md) and [Launch Demo](/examples/index.md#cross-functional-coordination).

## Agent Architectures

*   **BDI (Belief-Desire-Intention):** ...
*   **OODA (Observe-Orient-Decide-Act):** ...

## Base Classes

*   `BaseAgent`: ...