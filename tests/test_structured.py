import pytest
from pydantic import BaseModel

from crew.llm import LLMResult, Usage
from crew.models import Plan
from crew.structured import StructuredOutputError, extract_json, generate


class Thing(BaseModel):
    name: str
    n: int


class Scripted:
    model = "s"

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    async def complete(self, *, prompt, **kw):
        self.prompts.append(prompt)
        return LLMResult(self.replies.pop(0), Usage(1, 1))


async def test_valid_first_try_no_repair():
    llm = Scripted('{"name": "a", "n": 1}')
    t = await generate(llm, Thing, role="r", system="", prompt="p")
    assert t.n == 1 and len(llm.prompts) == 1


async def test_repairs_malformed_json_and_shows_error_to_model():
    llm = Scripted('{"name": "a", "n": 1,}', '```json\n{"name": "a", "n": 2}\n```')
    t = await generate(llm, Thing, role="r", system="", prompt="p")
    assert t.n == 2 and len(llm.prompts) == 2
    assert "invalid" in llm.prompts[1] and '"n": 1,}' in llm.prompts[1]


async def test_repairs_schema_violation():
    llm = Scripted('{"name": "a", "n": "many"}', '{"name": "a", "n": 3}')
    t = await generate(llm, Thing, role="r", system="", prompt="p")
    assert t.n == 3 and "n:" in llm.prompts[1]


async def test_gives_up_after_max_repairs():
    llm = Scripted("nope", "still nope", "no")
    with pytest.raises(StructuredOutputError):
        await generate(llm, Thing, role="r", system="", prompt="p", max_repairs=2)
    assert len(llm.prompts) == 3


def test_extract_json_handles_prose_and_fences():
    assert extract_json('Sure!\n{"a": 1}\nHope that helps') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}


def test_plan_rejects_cycles_and_unknown_deps():
    sub = lambda i, d: {"id": i, "question": "a long enough question", "depends_on": d}  # noqa: E731
    with pytest.raises(ValueError, match="->"):
        Plan.model_validate({"subtasks": [sub("a", ["b"]), sub("b", ["a"])]})
    with pytest.raises(ValueError, match="unknown"):
        Plan.model_validate({"subtasks": [sub("a", ["q"])]})
