import marimo

__generated_with = "0.13.4"
app = marimo.App(width="full")


@app.cell
def _():
    # Imports for types and standard libraries
    import asyncio
    import json

    # Configure logging
    import logging
    import pathlib
    import random
    import time
    import uuid
    from collections import defaultdict
    from collections.abc import Awaitable, Callable
    from datetime import datetime, timedelta
    from enum import Enum
    from typing import Any, Optional

    import marimo as mo
    import numpy as np
    import pandas as pd

    from agents.coordinators.product_launch import ProductLaunchCoordinator
    from agents.cross_functional import (
        CustomerServiceAgent,
        MarketingAgent,
        PricingAgent,
        StoreOpsAgent,
        SupplyChainAgent,
    )
    from agents.messaging import MessageBroker
    from agents.protocols.auction import ProcurementAuction
    from agents.protocols.contract_net import RetailCoordinator
    from agents.protocols.inventory_sharing import InventoryCollaborationNetwork
    from agents.store import StoreAgent

    # Import refactored components from project modules
    from models.messaging import AgentMessage, Performative
    from models.store import Store
    from models.supplier import Supplier, SupplierRating
    from models.task import Bid, Task, TaskStatus, TaskType
    from utils.planning import calculate_remediation_timeline

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    # Return all imported names needed by subsequent cells
    return (
        AgentMessage,
        Performative,
        MessageBroker,
        Task,
        TaskStatus,
        TaskType,
        Bid,
        StoreAgent,
        RetailCoordinator,
        Supplier,
        SupplierRating,
        ProcurementAuction,
        Store,
        InventoryCollaborationNetwork,
        CustomerServiceAgent,
        MarketingAgent,
        PricingAgent,
        StoreOpsAgent,
        SupplyChainAgent,
        ProductLaunchCoordinator,
        calculate_remediation_timeline,
        asyncio,
        datetime,
        timedelta,
        Enum,
        Any,
        Optional,
        list,
        dict,
        set,
        Callable,
        Awaitable,
        defaultdict,
        json,
        random,
        uuid,
        mo,
        pathlib,
        pd,
        np,
        time,
        logging,
        logger,
        # Add any other necessary models if used below, e.g. from procurement
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Chapter 8: Multi-Agent Systems in Retail

        This chapter dives into the collaborative power of Multi-Agent Systems
        (MAS) within retail environments. You'll explore specialized agent roles,
        orchestration patterns, and governance frameworks that enable these
        intelligent teams to work together seamlessly. Learn how multiple AI agents
        can coordinate to tackle complex retail challenges through practical
        examples and strategic implementation approaches.
        """
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Agent Communication Framework

        A robust communication framework is essential for MAS. This section shows
        a FIPA-inspired communication framework for retail agents, highlighting
        direct messaging, subscriptions, and conversation handling.
        """
    )
    return


@app.cell
def _(mo):
    # Define a state variable to hold the logs from the demo
    communication_logs = mo.state([])

    # Button to trigger the communication demo
    run_comm_button = mo.ui.button(label="Run Communication Demo")

    return communication_logs, run_comm_button


@app.cell
def _(asyncio, communication_logs, mo, run_comm_button):
    # Import the demo function
    from demos.agent_communication_demo import demo_retail_agent_communication

    _ = run_comm_button._on_click(lambda: communication_logs.set_value([]))

    async def run_and_log_communication():
        """Runs the demo and captures logs to state."""
        # In a real Marimo app, capturing stdout/logs requires more setup,
        # but the demo function now prints directly.
        # For now, we just run it.
        await demo_retail_agent_communication()
        # Update state after completion (maybe with a success message)
        communication_logs.set_value(["Communication Demo Completed (check console for logs)."])

    _ = run_comm_button._on_click(lambda: asyncio.create_task(run_and_log_communication()))

    mo.md(
        f"""
        {run_comm_button}

        **Output Log:**
        ```
        {{communication_logs.value}}
        ```
        """
    )
    return demo_retail_agent_communication, run_and_log_communication


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Task Allocation: Contract Net Protocol (CNP)

        CNP is a negotiation protocol where a manager (coordinator) announces a
        task, agents bid, and the coordinator awards the task to the best bidder.
        It's useful for decentralized resource allocation.
        """
    )
    return


@app.cell
def _(mo):
    cnp_logs = mo.state([])
    run_cnp_button = mo.ui.button(label="Run CNP Demo")
    return cnp_logs, run_cnp_button


@app.cell
def _(asyncio, cnp_logs, mo, run_cnp_button):
    # Import the demo function
    from demos.task_allocation_cnp_demo import demo_contract_net_protocol

    _ = run_cnp_button._on_click(lambda: cnp_logs.set_value([]))

    async def run_and_log_cnp():
        """Runs the CNP demo."""
        await demo_contract_net_protocol()
        cnp_logs.set_value(["CNP Demo Completed (check console for logs)."])

    _ = run_cnp_button._on_click(lambda: asyncio.create_task(run_and_log_cnp()))

    mo.md(
        f"""
        {run_cnp_button}

        **Output Log:**
        ```
        {{cnp_logs.value}}
        ```
        """
    )
    return demo_contract_net_protocol, run_and_log_cnp


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Supplier Selection: Procurement Auction

        Auctions (like reverse auctions for procurement) allow agents to compete
        based on price, delivery time, quality, etc. The system awards the
        contract based on predefined scoring criteria.
        """
    )
    return


@app.cell
def _(mo):
    auction_logs = mo.state([])
    run_auction_button = mo.ui.button(label="Run Auction Demo")
    return auction_logs, run_auction_button


@app.cell
def _(asyncio, auction_logs, mo, run_auction_button):
    # Import the demo function
    from demos.procurement_auction_demo import demo_procurement_auction

    _ = run_auction_button._on_click(lambda: auction_logs.set_value([]))

    async def run_and_log_auction():
        """Runs the Auction demo."""
        await demo_procurement_auction()
        auction_logs.set_value(["Auction Demo Completed (check console for logs)."])

    _ = run_auction_button._on_click(lambda: asyncio.create_task(run_and_log_auction()))

    mo.md(
        f"""
        {run_auction_button}

        **Output Log:**
        ```
        {{auction_logs.value}}
        ```
        """
    )
    return demo_procurement_auction, run_and_log_auction


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Inventory Management: Collaborative Sharing

        Agents representing different stores can collaborate to share inventory,
        transferring stock based on needs, excess, and transfer costs to optimize
        overall stock levels across the network.
        """
    )
    return


@app.cell
def _(mo):
    sharing_logs = mo.state([])
    run_sharing_button = mo.ui.button(label="Run Inventory Sharing Demo")
    return run_sharing_button, sharing_logs


@app.cell
def _(asyncio, mo, run_sharing_button, sharing_logs):
    # Import the demo function
    from demos.inventory_sharing_demo import demo_collaborative_inventory_sharing

    _ = run_sharing_button._on_click(lambda: sharing_logs.set_value([]))

    async def run_and_log_sharing():
        """Runs the Inventory Sharing demo."""
        await demo_collaborative_inventory_sharing()
        sharing_logs.set_value(["Inventory Sharing Demo Completed (check console for logs)."])

    _ = run_sharing_button._on_click(lambda: asyncio.create_task(run_and_log_sharing()))

    mo.md(
        f"""
        {run_sharing_button}

        **Output Log:**
        ```
        {{sharing_logs.value}}
        ```
        """
    )
    return demo_collaborative_inventory_sharing, run_and_log_sharing


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Cross-Functional Coordination: Product Launch

        A coordinator agent orchestrates tasks across multiple functional agents
        (Supply Chain, Pricing, Marketing, Store Ops, Customer Service) to ensure
        a synchronized and successful product launch.
        """
    )
    return


@app.cell
def _(mo):
    launch_logs = mo.state([])
    run_launch_button = mo.ui.button(label="Run Product Launch Demo")
    return launch_logs, run_launch_button


@app.cell
def _(asyncio, launch_logs, mo, run_launch_button):
    # Import the demo function
    from demos.product_launch_demo import demo_product_launch

    _ = run_launch_button._on_click(lambda: launch_logs.set_value([]))

    async def run_and_log_launch():
        """Runs the Product Launch demo."""
        await demo_product_launch()
        launch_logs.set_value(["Product Launch Demo Completed (check console for logs)."])

    _ = run_launch_button._on_click(lambda: asyncio.create_task(run_and_log_launch()))

    mo.md(
        f"""
        {run_launch_button}

        **Output Log:**
        ```
        {{launch_logs.value}}
        ```
        """
    )
    return demo_product_launch, run_and_log_launch


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary: Multi-Agent Systems in Retail

    This notebook has demonstrated several key aspects of multi-agent systems 
    in retail:

    1. **Agent Communication Protocols** - FIPA-inspired messaging framework 
       with performatives, message structure, and broker to facilitate direct 
       messaging and publish-subscribe patterns.
    2. **Contract Net Protocol** - Task allocation approach where a coordinator 
       announces tasks and store agents bid based on capacity, location, and 
       efficiency factors.
    3. **Auction Mechanisms** - Supplier selection process using a sealed-bid 
       reverse auction considering multiple attributes (price, delivery time, 
       quality).
    4. **Inventory Collaboration** - Coordinated inventory sharing between 
       stores based on needs, excess, and mutual benefit calculations.
    5. **Cross-Functional Workflow** - Product launch coordination across 
       multiple specialized departments working toward a unified goal.

    These patterns represent fundamental coordination approaches for complex 
    retail agent ecosystems where multiple specialized agents need to work 
    together effectively.

    Each demo highlights different aspects of multi-agent coordination:

    - **Communication** focuses on message exchange patterns
    - **Task Allocation** shows how to distribute workload efficiently
    - **Auction** demonstrates competitive resource allocation
    - **Inventory Sharing** illustrates cooperative resource optimization
    - **Product Launch** demonstrates cross-functional collaboration

    Together, these mechanisms form the foundation for building sophisticated 
    retail agent systems that can handle complex, distributed decision-making 
    challenges.
    """
    )
    return


if __name__ == "__main__":
    app.run()
