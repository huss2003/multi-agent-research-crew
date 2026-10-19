"""LLM interface + Anthropic implementation (structured output via forced tool use)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

PRICES_PER_MTOK = {"claude-sonnet-5-5": (3.0, 15.0), "mock-1": (3.0, 15.0)}
DEFAULT_PRICE = (3.0, 15.0)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __iadd__(self, o: Usage) -> Usage:
        self.input_tokens += o.input_tokens
        self.output_tokens += o.output_tokens
        return self

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResult:
    text: str
    usage: Usage


class LLM(Protocol):
    model: str

    async def complete(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResult:
        """Return raw text. With `schema`, text is the JSON the model produced (unvalidated)."""
        ...


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self.client = AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: type[BaseModel] | None = None,
        max_tokens: int = 1500,
    ) -> LLMResult:
        kw: dict = {}
        if schema is not None:
            kw["tools"] = [
                {
                    "name": "emit",
                    "description": f"Return the {schema.__name__} result.",
                    "input_schema": schema.model_json_schema(),
                }
            ]
            kw["tool_choice"] = {"type": "tool", "name": "emit"}
        msg = await self.client.messages.create(
            model=self.model,
            system=system,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kw,
        )
        text = ""
        for block in msg.content:
            if block.type == "tool_use":
                text = json.dumps(block.input)
                break
            if block.type == "text":
                text += block.text
        return LLMResult(text, Usage(msg.usage.input_tokens, msg.usage.output_tokens))
