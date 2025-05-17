from models.inventory import InventoryPosition, InventoryStatus
from models.store import Store


def test_inventory_position_methods():
    pos = InventoryPosition(product_id="P1", current_stock=50, target_stock=100, daily_sales_rate=10)
    assert pos.get_status() == InventoryStatus.LOW
    assert pos.needed_units() == 50
    assert pos.excess_units() == 0
    assert pos.days_of_supply() == 5

    pos.current_stock = 150
    assert pos.get_status() == InventoryStatus.EXCESS
    assert pos.excess_units() == 50
    assert pos.needed_units() == 0


def test_store_inventory_operations():
    store = Store(store_id="S1", name="Test Store", location="Loc")
    store.add_product("P1", current_stock=30, target_stock=100, sales_rate_per_day=5)
    store.add_product("P2", current_stock=160, target_stock=100, sales_rate_per_day=8)

    # update sales rate
    store.update_sales_rate("P1", 6)
    assert store.inventory["P1"].daily_sales_rate == 6

    # status checks
    assert store.get_inventory_status("P1") == InventoryStatus.LOW
    assert store.get_inventory_status("P2") == InventoryStatus.EXCESS

    # sharable/needed inventory
    assert store.get_sharable_inventory() == {"P2": 60}
    assert store.get_needed_inventory() == {"P1": 70}

    # transfer checks
    assert store.can_transfer("P2", 50)
    assert not store.can_transfer("P2", 70)

    # execute a transfer sending out
    assert store.execute_transfer("P2", 40, partner_id="S2", is_sending=True)
    assert store.inventory["P2"].current_stock == 120
    assert store.transfer_history[-1]["direction"] == "out"

    # receive inventory
    assert store.execute_transfer("P1", 20, partner_id="S2", is_sending=False)
    assert store.inventory["P1"].current_stock == 50
    assert store.transfer_history[-1]["direction"] == "in"

    # calculate transfer value
    value_send = store.calculate_transfer_value("P2", 10, is_sending=True)
    assert value_send > 0  # positive since stock high
    value_receive = store.calculate_transfer_value("P1", 10, is_sending=False)
    assert value_receive > 0  # valuable since low stock
