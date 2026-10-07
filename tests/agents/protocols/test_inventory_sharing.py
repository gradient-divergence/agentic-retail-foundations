import asyncio

import pytest

from agents.protocols.inventory_sharing import InventoryCollaborationNetwork
from models.store import Store


@pytest.mark.parametrize("quantity,receiver_has_product", [(5, True), (50, True), (5, False), (-5, True)])
def test_transfers_conserve_inventory_even_when_rejected(quantity, receiver_has_product):
    network = InventoryCollaborationNetwork()
    sender, receiver = Store("sender", "Sender", "A"), Store("receiver", "Receiver", "B")
    sender.add_product("sku", 20, 10, 1)
    if receiver_has_product:
        receiver.add_product("sku", 2, 10, 1)
    network.register_store(sender)
    network.register_store(receiver)
    before = sum(s.inventory["sku"].current_stock for s in (sender, receiver) if "sku" in s.inventory)
    operation = {"sender_id": "sender", "receiver_id": "receiver", "product_id": "sku", "quantity": quantity}
    result = asyncio.run(network.execute_transfers([operation]))[0]
    after = sum(s.inventory["sku"].current_stock for s in (sender, receiver) if "sku" in s.inventory)
    assert after == before
    assert result["status"] == ("completed" if quantity == 5 and receiver_has_product else "failed")
    if result["status"] == "failed":
        assert sender.inventory["sku"].current_stock == 20
        assert sender.transfer_history == receiver.transfer_history == []


def test_unknown_store_transfer_is_rejected_without_mutation():
    network = InventoryCollaborationNetwork()
    result = asyncio.run(
        network.execute_transfers(
            [{"sender_id": "missing", "receiver_id": "missing", "product_id": "sku", "quantity": 1}]
        )
    )
    assert result[0]["status"] == "failed"
