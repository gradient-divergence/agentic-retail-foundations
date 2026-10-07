import pytest

from models.store import Store


@pytest.mark.parametrize("quantity", [-5, 0])
@pytest.mark.parametrize("is_sending", [True, False])
def test_nonpositive_transfers_never_mutate_inventory(quantity, is_sending):
    store = Store("S", "Store", "Toronto")
    store.add_product("A", current_stock=20, target_stock=10, sales_rate_per_day=1)
    assert not store.can_transfer("A", quantity)
    assert not store.execute_transfer("A", quantity, "other", is_sending)
    assert store.inventory["A"].current_stock == 20
    assert store.transfer_history == []
