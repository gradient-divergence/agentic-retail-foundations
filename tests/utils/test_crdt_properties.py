import copy
import itertools
import random

import pytest

from utils.crdt import PNCounter


def merged(left, right):
    result = copy.deepcopy(left)
    result.merge(right)
    return result


def test_random_merge_laws_and_operation_orders():
    rng = random.Random(20261002)
    for _ in range(50):
        replicas = [PNCounter("sku", "store", initial_value=10) for _ in range(3)]
        operations = [
            (rng.randrange(3), rng.choice(["increment", "decrement"]), rng.randint(1, 9)) for _ in range(30)
        ]
        rng.shuffle(operations)
        for node, operation, quantity in operations:
            getattr(replicas[node], operation)(str(node), quantity)
        a, b, c = replicas
        assert merged(a, b).state == merged(b, a).state
        assert merged(merged(a, b), c).state == merged(a, merged(b, c)).state
        assert merged(a, a).state == a.state
        expected = 10 + sum(q if op == "increment" else -q for _, op, q in operations)
        for order in itertools.permutations(replicas):
            result = PNCounter("sku", "store")
            for replica in order:
                result.merge(replica)
                result.merge(replica)
            assert result.value() == expected


def test_restored_replica_and_snapshots_are_independent():
    original = PNCounter("sku", "store", initial_value=10)
    snapshot = original.to_dict()
    restored = PNCounter.from_dict(snapshot)
    restored.increment("initial", 5)
    restored.decrement("store", 2)
    assert original.value() == 10
    assert snapshot["increments"] == {"initial": 10}
    state = original.state
    state["p"]["initial"] = 999
    assert original.value() == 10


@pytest.mark.parametrize("product,location", [("other", "store"), ("sku", "other")])
def test_merge_rejects_different_inventory_keys(product, location):
    counter = PNCounter("sku", "store", initial_value=10)
    with pytest.raises(ValueError):
        counter.merge(PNCounter(product, location, initial_value=20))
    assert counter.value() == 10
