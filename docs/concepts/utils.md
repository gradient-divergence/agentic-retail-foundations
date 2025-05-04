# Utilities

This section describes reusable utility classes and functions found primarily in the `utils/` directory, supporting various agent operations.

-   **`event_bus.py`:** Implements an asynchronous `EventBus` for pub/sub communication. See [API Reference](../reference/utils.md#utils.event_bus.EventBus).

-   **`crdt.py`:** Contains `PNCounter`, a Positive-Negative Counter CRDT for distributed state tracking. See [API Reference](../reference/utils.md#utils.crdt.PNCounter).

-   **`monitoring.py`:** Provides `AgentMonitor` for tracking metrics and detecting drift. See [API Reference](../reference/utils.md#utils.monitoring.AgentMonitor).

-   **`planning.py`:** Includes fulfillment optimization components:
    -   `StoreLayout`: Represents store physical grid. See [API Reference](../reference/utils.md#utils.planning.StoreLayout).
    -   `FulfillmentPlanner`: Assigns orders and generates paths. See [API Reference](../reference/utils.md#utils.planning.FulfillmentPlanner).
    -   `calculate_remediation_timeline`: Helper for estimating delays. See [API Reference](../reference/utils.md#utils.planning.calculate_remediation_timeline).

-   **`nlp.py`:** NLP helper functions using LLMs for tasks like intent classification. See [API Reference](../reference/utils.md#nlp-helpers).

-   **`openai_utils.py`:** Contains `safe_chat_completion` wrapper for robust OpenAI calls. See [API Reference](../reference/utils.md#utils.openai_utils.safe_chat_completion).

-   **`data_generation.py`:** Functions for creating synthetic data.

-   **`logger.py`:** Standardized logger setup.

-   **`env.py`:** `.env` file loading helper.

Refer to the [Utilities API Reference](/reference/utils.md) for detailed class/function signatures. 