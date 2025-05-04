# Models API Reference

This page provides auto-generated API documentation for the core data model classes.

## Communication

::: models.messaging.AgentMessage
    options:
      show_root_heading: true
      show_source: false

## Fulfillment

::: models.fulfillment.Order
    options:
      show_root_heading: true
      show_source: false
      members:
        - add_event
        - update_status

::: models.fulfillment.OrderLineItem
    options:
      show_root_heading: true
      show_source: false

::: models.fulfillment.Item
    options:
      show_root_heading: true
      show_source: false

::: models.fulfillment.Associate
    options:
      show_root_heading: true
      show_source: false

## Inventory & Store

::: models.inventory.InventoryPosition
    options:
      show_root_heading: true
      show_source: false
      members:
        - get_status
        - excess_units
        - needed_units
        - days_of_supply

::: models.store.Store
    options:
      show_root_heading: true
      show_source: false
      members:
        - add_product
        - update_sales_rate
        - get_inventory_status
        - get_sharable_inventory
        - get_needed_inventory
        - can_transfer
        - execute_transfer
        - calculate_transfer_value

## Procurement & Supplier

::: models.procurement.PurchaseOrder
    options:
      show_root_heading: true
      show_source: false

::: models.procurement.SupplierBid
    options:
      show_root_heading: true
      show_source: false

::: models.supplier.Supplier
    options:
      show_root_heading: true
      show_source: false
      members:
        - can_supply

## Task Management

::: models.task.Task
    options:
      show_root_heading: true
      show_source: false

::: models.task.Bid
    options:
      show_root_heading: true
      show_source: false

## Events & State

::: models.events.RetailEvent
    options:
      show_root_heading: true
      show_source: false

::: models.events.InventoryEvent
    options:
      show_root_heading: true
      show_source: false

::: models.state.ProductInventoryState
    options:
      show_root_heading: true
      show_source: false
      members:
        - calculate_available

::: models.state.InventoryReservation
    options:
      show_root_heading: true
      show_source: false

## API Models

::: models.api.Agent
    options:
      show_root_heading: true
      show_source: false

::: models.api.Token
    options:
      show_root_heading: true
      show_source: false

::: models.api.RequestLogEntry
    options:
      show_root_heading: true
      show_source: false

*(Note: Enums are typically documented via their usage or in concepts, but could be added here too)*