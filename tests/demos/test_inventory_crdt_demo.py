import ast
import copy
import random
from pathlib import Path

import pytest
from pydantic import BaseModel

# Compile the actual pure listing classes; optional server imports are excluded.
source = Path(__file__).resolve().parents[2] / "demos/inventory_crdt_demo.py"
tree = ast.parse(source.read_text())
definitions = [
    node
    for node in tree.body
    if (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    or (isinstance(node, ast.ClassDef) and node.name in {"PNCounter", "PNCounterSnapshot"})
]
namespace = {"BaseModel": BaseModel}
exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
PNCounter, PNCounterSnapshot = namespace["PNCounter"], namespace["PNCounterSnapshot"]


def test_random_book_crdt_merge_laws():
    rng = random.Random(20261002)
    for _ in range(50):
        counters = [PNCounter("sku", "store", 10) for _ in range(3)]
        for index, counter in enumerate(counters):
            for _ in range(10):
                getattr(counter, rng.choice(["increment", "decrement"]))(str(index), rng.randint(1, 9))
        a, b, c = counters
        assert a.merge(b).to_dict() == b.merge(a).to_dict()
        assert a.merge(b).merge(c).to_dict() == a.merge(b.merge(c)).to_dict()
        assert a.merge(a).to_dict() == a.to_dict()
        before = copy.deepcopy(a.to_dict())
        rng.shuffle(counters)
        result = counters[0].merge(counters[1]).merge(counters[2])
        assert result.to_dict() == a.merge(b).merge(c).to_dict()
        assert a.to_dict() == before


def test_snapshot_does_not_share_state_with_restored_counter():
    snapshot = PNCounterSnapshot(product_id="sku", location_id="store", increments={"a": 10}, decrements={})
    restored = PNCounter.from_dict(snapshot)
    restored.increment("a", 5)
    assert snapshot.increments == {"a": 10}


def test_book_crdt_rejects_other_inventory_keys():
    with pytest.raises(ValueError):
        PNCounter("sku", "store").merge(PNCounter("sku", "other"))


def test_server_imports():
    from demos import inventory_crdt_demo

    assert inventory_crdt_demo.PNCounter("sku", "store", 10).value() == 10
