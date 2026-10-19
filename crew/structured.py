"""Validated structured output with an LLM repair loop."""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from crew.llm import LLM
from crew.tracing import Tracer

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


def extract_json(text: str) -> object:
    """Parse JSON from raw model text, tolerating code fences and surrounding prose."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


async def generate(
    llm: LLM,
    schema: type[T],
    *,
    role: str,
    system: str,
    prompt: str,
    max_tokens: int = 1500,
    max_repairs: int = 2,
    tracer: Tracer | None = None,
    agent: str = "",
) -> T:
    """Call the model, validate against `schema`; on failure show the model its own output
    and the validation error and ask again (up to `max_repairs` times)."""
    current, last_err = prompt, ""
    for attempt in range(max_repairs + 1):
        res = await llm.complete(
            role=role, system=system, prompt=current, schema=schema, max_tokens=max_tokens
        )
        try:
            return schema.model_validate(extract_json(res.text))
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = _short(e)
            if tracer:
                tracer.emit("repair", agent, role=role, attempt=attempt + 1, error=last_err[:200])
            current = (
                f"{prompt}\n\nYour previous reply was invalid.\nREPLY:\n{res.text[:2000]}\n"
                f"ERROR:\n{last_err}\nReturn ONLY corrected JSON matching the schema."
            )
    raise StructuredOutputError(f"{role}: invalid output after {max_repairs} repairs: {last_err}")


def _short(e: Exception) -> str:
    if isinstance(e, ValidationError):
        return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])
    return f"JSON parse error: {e}"
