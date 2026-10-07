import asyncio

from connectors.dummy_db import DummyDB


def test_missing_customer_is_reported():
    assert asyncio.run(DummyDB().get_customer("missing")) is None


def test_blank_identifier_does_not_resolve_to_first_item():
    assert asyncio.run(DummyDB().resolve_product_id("")) is None
