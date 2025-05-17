import pytest

from utils.crdt import PNCounter

# --- Test Initialization --- #


def test_pncounter_init_default():
    """Test PNCounter initialization with default value (0)."""
    counter = PNCounter(product_id="prod1", location_id="locA")
    assert counter.product_id == "prod1"
    assert counter.location_id == "locA"
    assert counter.increments == {}
    assert counter.decrements == {}
    assert counter.value() == 0


def test_pncounter_init_positive():
    """Test PNCounter initialization with a positive initial value."""
    counter = PNCounter(product_id="prod2", location_id="locB", initial_value=10)
    assert counter.increments == {"initial": 10}
    assert counter.decrements == {}
    assert counter.value() == 10


def test_pncounter_init_negative():
    """Test PNCounter initialization with a negative initial value."""
    counter = PNCounter(product_id="prod3", location_id="locC", initial_value=-5)
    assert counter.increments == {}
    assert counter.decrements == {"initial": 5}
    assert counter.value() == -5


# --- Test Increment --- #


def test_pncounter_increment():
    """Test basic increment operations."""
    counter = PNCounter("p1", "l1")

    # Increment node A by 1
    counter.increment("nodeA")
    assert counter.increments == {"nodeA": 1}
    assert counter.decrements == {}
    assert counter.value() == 1

    # Increment node A again by 2
    counter.increment("nodeA", value=2)
    assert counter.increments == {"nodeA": 3}
    assert counter.value() == 3

    # Increment node B by 5
    counter.increment("nodeB", 5)
    assert counter.increments == {"nodeA": 3, "nodeB": 5}
    assert counter.value() == 8


def test_pncounter_increment_invalid_value():
    """Test incrementing with zero or negative value raises ValueError."""
    counter = PNCounter("p1", "l1")
    with pytest.raises(ValueError, match="Increment value must be positive"):
        counter.increment("nodeA", 0)
    with pytest.raises(ValueError, match="Increment value must be positive"):
        counter.increment("nodeB", -1)
    # Ensure state didn't change
    assert counter.increments == {}
    assert counter.value() == 0


# --- Test Decrement --- #


def test_pncounter_decrement():
    """Test basic decrement operations."""
    counter = PNCounter("p1", "l1")

    # Decrement node A by 1
    counter.decrement("nodeA")
    assert counter.decrements == {"nodeA": 1}
    assert counter.increments == {}
    assert counter.value() == -1

    # Decrement node A again by 3
    counter.decrement("nodeA", value=3)
    assert counter.decrements == {"nodeA": 4}
    assert counter.value() == -4

    # Decrement node B by 2
    counter.decrement("nodeB", 2)
    assert counter.decrements == {"nodeA": 4, "nodeB": 2}
    assert counter.value() == -6


def test_pncounter_decrement_invalid_value():
    """Test decrementing with zero or negative value raises ValueError."""
    counter = PNCounter("p1", "l1")
    with pytest.raises(ValueError, match="Decrement value must be positive"):
        counter.decrement("nodeA", 0)
    with pytest.raises(ValueError, match="Decrement value must be positive"):
        counter.decrement("nodeB", -1)
    # Ensure state didn't change
    assert counter.decrements == {}
    assert counter.value() == 0


# --- Test value() --- #


def test_pncounter_value_calculation():
    """Test the value() calculation with mixed increments and decrements."""
    counter = PNCounter("prodX", "locX", initial_value=5)  # Start at 5 (initial: p=5)

    # inc(A, 3) -> p = {initial:5, A:3}
    counter.increment("nodeA", 3)
    assert counter.value() == 8

    # dec(B, 2) -> p = {initial:5, A:3}, n = {B:2}
    counter.decrement("nodeB", 2)
    assert counter.value() == 6  # 8 - 2

    # inc(B, 1) -> p = {initial:5, A:3, B:1}, n = {B:2}
    counter.increment("nodeB", 1)
    assert counter.value() == 7  # 9 - 2

    # dec(A, 4) -> p = {initial:5, A:3, B:1}, n = {B:2, A:4}
    counter.decrement("nodeA", 4)
    assert counter.value() == 3  # 9 - 6

    # dec(C, 1) -> p = {initial:5, A:3, B:1}, n = {B:2, A:4, C:1}
    counter.decrement("nodeC", 1)
    assert counter.value() == 2  # 9 - 7

    # inc(A, 1) -> p = {initial:5, A:4, B:1}, n = {B:2, A:4, C:1} # A updated
    counter.increment("nodeA", 1)
    assert counter.value() == 3  # 10 - 7


# --- Test merge() --- #


def test_pncounter_merge():
    """Test merging two PNCounters."""
    # Counter 1 state
    c1 = PNCounter("p1", "l1")
    c1.increment("A", 5)
    c1.increment("B", 2)
    c1.decrement("X", 3)
    c1.decrement("Y", 6)
    # c1 state: p={A:5, B:2}, n={X:3, Y:6}, value = 7 - 9 = -2
    assert c1.value() == -2

    # Counter 2 state (overlapping and unique nodes)
    c2 = PNCounter("p1", "l1")
    c2.increment("A", 3)  # Lower increment for A
    c2.increment("C", 4)  # Unique increment for C
    c2.decrement("X", 5)  # Higher decrement for X
    c2.decrement("Z", 1)  # Unique decrement for Z
    # c2 state: p={A:3, C:4}, n={X:5, Z:1}, value = 7 - 6 = 1
    assert c2.value() == 1

    # Store c2 state before merge to check it doesn't change
    c2_inc_orig = c2.increments.copy()
    c2_dec_orig = c2.decrements.copy()
    c2_val_orig = c2.value()

    # Merge c2 into c1
    c1.merge(c2)

    # Expected merged state in c1 (taking max for each node)
    expected_increments = {
        "A": 5,  # max(c1[A]=5, c2[A]=3)
        "B": 2,  # max(c1[B]=2, c2[B]=0)
        "C": 4,  # max(c1[C]=0, c2[C]=4)
    }
    expected_decrements = {
        "X": 5,  # max(c1[X]=3, c2[X]=5)
        "Y": 6,  # max(c1[Y]=6, c2[Y]=0)
        "Z": 1,  # max(c1[Z]=0, c2[Z]=1)
    }
    expected_value = sum(expected_increments.values()) - sum(expected_decrements.values())
    # (5+2+4) - (5+6+1) = 11 - 12 = -1

    assert c1.increments == expected_increments
    assert c1.decrements == expected_decrements
    assert c1.value() == expected_value

    # Verify c2 did not change
    assert c2.increments == c2_inc_orig
    assert c2.decrements == c2_dec_orig
    assert c2.value() == c2_val_orig


def test_pncounter_merge_idempotent():
    """Test that merging the same counter multiple times has no extra effect."""
    c1 = PNCounter("p1", "l1")
    c1.increment("A", 5)
    c1.decrement("X", 3)

    c2 = PNCounter("p1", "l1")
    c2.increment("A", 3)
    c2.increment("B", 2)
    c2.decrement("X", 5)

    # First merge
    c1.merge(c2)
    state_after_first_merge_p = c1.increments.copy()
    state_after_first_merge_n = c1.decrements.copy()
    value_after_first_merge = c1.value()

    # Second merge (should have no effect)
    c1.merge(c2)

    assert c1.increments == state_after_first_merge_p
    assert c1.decrements == state_after_first_merge_n
    assert c1.value() == value_after_first_merge


def test_pncounter_merge_type_error():
    """Test merging with a non-PNCounter raises TypeError."""
    c1 = PNCounter("p1", "l1")
    not_a_counter = {"p": {}, "n": {}}

    with pytest.raises(TypeError, match="Can only merge with another PNCounter"):
        c1.merge(not_a_counter)  # type: ignore [arg-type]


# --- Test State / Serialization --- #


def test_pncounter_state_property():
    """Test the state property returns the correct format."""
    counter = PNCounter("p1", "l1")
    counter.increment("A", 3)
    counter.decrement("B", 2)

    expected_state = {"p": {"A": 3}, "n": {"B": 2}}
    assert counter.state == expected_state


def test_pncounter_to_from_dict():
    """Test serializing to dict and deserializing back."""
    # Create original counter
    counter_orig = PNCounter("prod_serial", "loc_serial", initial_value=1)
    counter_orig.increment("nodeS1", 4)
    counter_orig.decrement("nodeS2", 2)
    orig_value = counter_orig.value()

    # Serialize
    data = counter_orig.to_dict()

    # Check dict format
    assert data["product_id"] == "prod_serial"
    assert data["location_id"] == "loc_serial"
    assert data["increments"] == {"initial": 1, "nodeS1": 4}
    assert data["decrements"] == {"nodeS2": 2}

    # Deserialize
    counter_new = PNCounter.from_dict(data)

    # Check new counter state
    assert isinstance(counter_new, PNCounter)
    assert counter_new.product_id == counter_orig.product_id
    assert counter_new.location_id == counter_orig.location_id
    assert counter_new.increments == counter_orig.increments
    assert counter_new.decrements == counter_orig.decrements
    assert counter_new.value() == orig_value


def test_pncounter_from_dict_invalid_data():
    """Test from_dict raises ValueError for invalid data format."""
    invalid_data_missing_keys = {
        "product_id": "p1",
        # missing location_id, increments, decrements
    }
    with pytest.raises(ValueError, match="Invalid data format"):
        PNCounter.from_dict(invalid_data_missing_keys)


def test_pncounter_from_dict_invalid_types():
    """Test from_dict handles invalid types for increments/decrements."""
    data_with_bad_types = {
        "product_id": "p1",
        "location_id": "l1",
        "increments": [1, 2, 3],  # Should be dict
        "decrements": "not a dict",  # Should be dict
    }
    counter = PNCounter.from_dict(data_with_bad_types)
    # Should default to empty dicts
    assert counter.increments == {}
    assert counter.decrements == {}
    assert counter.value() == 0
