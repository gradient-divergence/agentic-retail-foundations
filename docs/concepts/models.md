# Core Concepts: Data Models

<!--
Describe the key data structures used in the project (e.g., Product, Inventory,
Customer, Sale). Explain how Pydantic is used for validation and structure.
Include examples of how these models are used by agents or in simulations.
-->

## Pydantic Models

We use Pydantic for robust data modeling. Key models include:

*   `Product`: ...
*   `Inventory`: ...
*   `Sale`: ...

# Data Models

This section provides an overview of the core data structures (models and enums) used throughout the agentic retail framework, primarily located in the `models/` directory.

These models define the structure of information exchanged between agents and used in their decision-making processes. Pydantic and Dataclasses are used for validation and type hinting.

-   **`enums.py`:** Defines various Enumerations (`Enum`) used for standardized values (e.g., `AgentType`, `OrderStatus`, `Performative`), promoting consistency and preventing errors.

-   **`messaging.py`:** Models for agent communication: `AgentMessage` (FIPA-inspired structure) and `Performative` (intent).

-   **`fulfillment.py`:** Models for order fulfillment: `Item` (physical product), `OrderLineItem` (product within an order), `Order` (customer order), and `Associate` (store worker).

-   **`inventory.py`:** `InventoryPosition` for tracking stock, target, and sales rate per product/location.

-   **`procurement.py`:** Models for purchasing: `PurchaseOrder` and `SupplierBid`.

-   **`store.py`:** The `Store` model, representing a location with inventory and collaboration logic.

-   **`supplier.py`:** Models for suppliers: `Supplier` and related enums.

-   **`task.py`:** Models for task allocation (CNP): `Task`, `Bid`, and status/type enums.

-   **`events.py`:** Pydantic models for event-driven systems: `RetailEvent`, `InventoryEvent`, etc.

-   **`state.py`:** Pydantic models for state representation: `ProductInventoryState`, `InventoryReservation`.

-   **`api.py`:** Contains Pydantic models used specifically in API definitions (e.g., API Gateway), such as `Token`, `TokenData`, `Agent` (for auth context), and `RequestLogEntry`.

Refer to the [Models API Reference](/reference/models.md) for detailed class signatures.