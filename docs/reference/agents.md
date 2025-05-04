# API Reference: Agents

<!--
This page will automatically generate documentation from your code's docstrings
using the mkdocstrings plugin. Add directives like the example below for each
public module, class, or function you want to document within the 'agents' package.
-->

::: agents
    options:
      show_root_heading: true
      show_source: false

<!-- Example: Document a specific module -->
::: agents.bdi

<!-- Example: Document a specific class 
::: agents.base.BaseAgent
-->

# Agents API Reference

This page provides auto-generated API documentation for the core agent classes.

## Base Agent

::: agents.base.BaseAgent
    options:
      show_root_heading: true
      show_source: false
      # Show key methods if any are added beyond init/publish/handle_exception
      # members:
      #  - some_important_method

## Core Agents

### LLM Customer Service

::: agents.llm.RetailCustomerServiceAgent
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - process_customer_inquiry
        # Exclude internal helpers like _classify_intent, _generate_response

### Master Orchestrator

::: agents.orchestrator.MasterOrchestrator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - register_event_handlers
        - handle_order_event
        - handle_exception_event
        # Exclude internal helpers like _check_for_stalled_orders

## Protocols & Coordinators

### Contract Net Coordinator

::: agents.protocols.contract_net.RetailCoordinator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - register_participant
        - announce_task
        - handle_bid
        - award_task
        - update_task_status

### Procurement Auction

::: agents.protocols.auction.ProcurementAuction
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - register_supplier
        - submit_bid
        - start_auction
        - advance_round
        - finalize_auction
        - cancel_auction

### Inventory Collaboration

::: agents.protocols.inventory_sharing.InventoryCollaborationNetwork
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - register_store
        - calculate_transfer_cost
        - identify_transfer_opportunities
        # Add execute_transfer when implemented

### Product Launch Coordinator

::: agents.coordinators.product_launch.ProductLaunchCoordinator
    options:
      show_root_heading: true
      show_source: false
      members:
        - __init__
        - coordinate_product_launch

*(Note: Other agent classes like QLearningAgent, BayesianAgent, etc., can be added here)*