from datetime import datetime

import pytest

from models.inventory import InventoryPosition, InventoryStatus


def test_last_updated_is_the_instance_creation_time():
    before = datetime.now()
    position = InventoryPosition("A", 0, 10, 1)
    assert before <= position.last_updated <= datetime.now()


@pytest.mark.parametrize("stock,status", [(0, InventoryStatus.ADEQUATE), (5, InventoryStatus.EXCESS)])
def test_zero_target_inventory_has_a_defined_status(stock, status):
    assert InventoryPosition("A", stock, 0, 1).get_status() == status
