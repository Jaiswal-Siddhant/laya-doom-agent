from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from laya_doom.actions.models import DoomAction
from laya_doom.observation.models import Observation


class StepResult(BaseModel):
    reward: float
    finished: bool


class DoomEnvironment(Protocol):
    def reset(self) -> None:
        ...

    def observe(self) -> Observation:
        ...

    def step(self, action: DoomAction) -> StepResult:
        ...

    def is_finished(self) -> bool:
        ...

    def close(self) -> None:
        ...
