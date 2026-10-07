import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from utils import nlp


def test_nlp_module_imports_without_sdk():
    import subprocess
    import sys

    code = """
import sys
sys.modules['openai'] = None
from utils import nlp
assert callable(nlp.classify_intent)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("reply", ["ORD999", "Here is ORD987"])
def test_inferred_order_id_must_be_a_recent_id(monkeypatch, reply):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])
    monkeypatch.setattr(nlp, "safe_chat_completion", AsyncMock(return_value=completion))
    result = asyncio.run(
        nlp.extract_order_id_llm(
            AsyncMock(spec=AsyncOpenAI),
            message="My order?",
            recent_order_ids=["ORD987"],
            model="fake",
            logger=logging.getLogger(__name__),
        )
    )
    assert result is None


def test_not_found_is_not_a_product_identifier(monkeypatch):
    completion = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not_found"))])
    monkeypatch.setattr(nlp, "safe_chat_completion", AsyncMock(return_value=completion))
    result = asyncio.run(
        nlp.extract_product_id(
            AsyncMock(),
            message="Store hours?",
            model="fake",
            logger=logging.getLogger(__name__),
        )
    )
    assert result is None
