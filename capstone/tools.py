from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

ToolOutput: TypeAlias = BaseModel | dict[str, JsonValue]


class ToolSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str
    permission: Literal["read", "write", "restricted"]
    schema_: dict[str, JsonValue] = Field(alias="schema")
    handler: Callable[..., ToolOutput] = Field(exclude=True)

    def execute(self, **kwargs: JsonValue) -> dict[str, JsonValue]:
        result = self.handler(**kwargs)
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result
