import asyncio
from unittest.mock import patch

from capstone.orchestrator import CapstoneOrchestrator
from demos import (
    agent_communication_demo,
    capstone_gateway_demo,
    capstone_scaffold_demo,
    inventory_sharing_demo_short,
    procurement_auction_demo,
    procurement_auction_demo_short,
    task_allocation_cnp_demo,
)
from models.task import TaskStatus


def test_communication_demo_output(capsys):
    asyncio.run(agent_communication_demo.demo_retail_agent_communication())
    assert capsys.readouterr().out.splitlines() == [
        "[inventory] P1001 stock=15",
        "[replenishment] ok stock: P1001=15",
        "[inventory] subscribed replenishment to topic 'inventory_alerts'",
        "[replenishment] low stock: P1002=3",
    ]


def test_procurement_demo_winners(capsys):
    asyncio.run(procurement_auction_demo_short.demo_procurement_auction())
    assert capsys.readouterr().out == "Winner: sup-beta price=$10800.00 days=9\n"
    asyncio.run(procurement_auction_demo.demo_procurement_auction())
    assert "Winner: Beta Goods Inc." in capsys.readouterr().out


def test_contract_net_demo_awards_lowest_cost_agent(capsys):
    async def execute(_agent, task):
        task.status = TaskStatus.COMPLETED
        return True

    with patch.object(task_allocation_cnp_demo.StoreAgent, "execute_task", execute):
        asyncio.run(task_allocation_cnp_demo.demo_contract_net_protocol())
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert all("awarded to North Store" in line for line in lines)


def test_inventory_demo_conserves_total():
    totals = []
    execute = inventory_sharing_demo_short.InventoryCollaborationNetwork.execute_transfers

    async def checked_execute(network, operations):
        def total():
            return sum(
                position.current_stock
                for store in network.stores.values()
                for position in store.inventory.values()
            )

        before = total()
        results = await execute(network, operations)
        totals.append((before, total()))
        assert all(result["status"] == "completed" for result in results)
        return results

    with patch.object(
        inventory_sharing_demo_short.InventoryCollaborationNetwork, "execute_transfers", checked_execute
    ):
        asyncio.run(inventory_sharing_demo_short.demo_collaborative_inventory_sharing())
    assert totals == [(215, 215)]


def test_capstone_demos_execute_valid_tools(capsys):
    capstone_scaffold_demo.main()
    assert "'recommended_action': 'replenish'" in capsys.readouterr().out
    asyncio.run(capstone_gateway_demo.main())
    assert "'status': 'reserved'" in capsys.readouterr().out


def test_unknown_tool_has_audit_and_never_executes():
    from capstone.policies import PolicyEvaluator
    from capstone.schemas import ToolCall
    from capstone.tracing import TraceContext

    result, audit = CapstoneOrchestrator([], PolicyEvaluator()).handle_tool_call(
        ToolCall(name="unknown", args={}), TraceContext(route="test")
    )
    assert result.output == {"error": "unknown_tool"}
    assert audit.decision == "unknown_tool"
