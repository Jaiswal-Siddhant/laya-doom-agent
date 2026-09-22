from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from laya_doom.actions.models import DoomAction
from laya_doom.observation.models import Observation


class AgentDecision(BaseModel):
    action: DoomAction
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float = Field(ge=0.0)
    raw_response: dict[str, Any] | None = None


class DecisionEngine(Protocol):
    def decide(self, observation: Observation) -> AgentDecision:
        ...
