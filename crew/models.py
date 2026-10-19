"""Pydantic contracts for every structured LLM output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from crew.orchestrator import topo_sort


class SubTask(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,23}$")
    question: str = Field(min_length=8)
    depends_on: list[str] = Field(default_factory=list)
    priority: int = Field(default=2, ge=1, le=3)  # 1 = must keep when degrading


class Plan(BaseModel):
    subtasks: list[SubTask] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def _is_dag(self) -> Plan:
        ids = [s.id for s in self.subtasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate subtask ids")
        topo_sort({s.id: s.depends_on for s in self.subtasks})  # unknown dep / cycle -> error
        return self


class Queries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=4)


class RawFinding(BaseModel):
    claim: str = Field(min_length=10)
    source_url: str
    quote: str = Field(min_length=20, max_length=500)


class Findings(BaseModel):
    findings: list[RawFinding] = Field(max_length=10)


class Statement(BaseModel):
    text: str = Field(min_length=5)
    evidence_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    heading: str
    statements: list[Statement] = Field(min_length=1)


class Draft(BaseModel):
    title: str
    sections: list[Section] = Field(min_length=1)


class Verdict(BaseModel):
    statement_id: int
    supported: bool
    reason: str = ""


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class VerificationRow(BaseModel):
    id: int
    section: str
    text: str
    evidence_ids: list[str]
    status: Literal["supported", "unsupported", "uncited"]
    reason: str = ""
