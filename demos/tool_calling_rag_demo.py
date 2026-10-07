#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Product(BaseModel):
    sku: str
    name: str
    category: str
    price: float


class SearchCatalogArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    query: str


class ReturnPolicyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchCatalogCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["search_catalog"]
    args: SearchCatalogArgs


class ReturnPolicyCall(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["get_return_policy"]
    args: ReturnPolicyArgs


ToolCall = SearchCatalogCall | ReturnPolicyCall


class AssistantResponse(BaseModel):
    intent: str
    answer: str
    sources: list[str]
    confidence: float = Field(..., ge=0, le=1)
    trace_id: str


CATALOG = [
    Product(
        sku="SKU-1001",
        name="Waterproof Trail Jacket",
        category="outerwear",
        price=129.0,
    ),
    Product(
        sku="SKU-2002",
        name="All-Terrain Running Shoes",
        category="footwear",
        price=89.0,
    ),
    Product(
        sku="SKU-3003",
        name="Insulated Thermal Bottle",
        category="accessories",
        price=24.0,
    ),
]


def search_catalog(query: str) -> list[Product]:
    query_lower = query.lower()
    return [p for p in CATALOG if query_lower in p.name.lower() or query_lower in p.category]


def get_return_policy() -> str:
    return "Returns accepted within 30 days with receipt. Worn items excluded."


# region book:tool-calling-demo
def route_query(query: str) -> ToolCall:
    if "return" in query.lower():
        return ReturnPolicyCall(name="get_return_policy", args=ReturnPolicyArgs())
    return SearchCatalogCall(name="search_catalog", args=SearchCatalogArgs(query=query))


def run_tool_call(query: str) -> AssistantResponse:
    trace_id = str(uuid4())
    tool_call = route_query(query)
    tool_call = TypeAdapter(ToolCall).validate_python(
        tool_call.model_dump() if isinstance(tool_call, BaseModel) else tool_call
    )

    if tool_call.name == "search_catalog":
        results = search_catalog(query=tool_call.args.query)
        answer = "Top matches: " + ", ".join(p.name for p in results)
        sources = [p.sku for p in results]
    else:
        answer = get_return_policy()
        sources = ["policy:returns"]

    return AssistantResponse(
        intent=tool_call.name,
        answer=answer,
        sources=sources,
        confidence=0.86,
        trace_id=trace_id,
    )


# endregion book:tool-calling-demo


def log_event(trace_id: str, status: str, latency_ms: int) -> None:
    payload = {
        "trace_id": trace_id,
        "route": "tool_calling_rag_demo",
        "status": status,
        "latency_ms": latency_ms,
    }
    print(json.dumps(payload))


def main() -> None:
    start = time.perf_counter()
    response = run_tool_call("running shoes")
    latency_ms = int((time.perf_counter() - start) * 1000)
    log_event(response.trace_id, "ok", latency_ms)
    print(response.model_dump())


if __name__ == "__main__":
    main()
