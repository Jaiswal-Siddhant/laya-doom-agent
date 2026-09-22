from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from laya_doom.actions.validator import ActionValidator
from laya_doom.decision.base import DecisionEngine
from laya_doom.environment.base import DoomEnvironment
from laya_doom.telemetry.events import TelemetryRecorder

logger = logging.getLogger(__name__)


class EpisodeResult(BaseModel):
    steps: int = Field(ge=0)
    total_reward: float


class DoomAgent:
    def __init__(
        self,
        environment: DoomEnvironment,
        decision_engine: DecisionEngine,
        action_validator: ActionValidator,
        telemetry: TelemetryRecorder,
        max_steps: int | None = None,
    ) -> None:
        self._environment = environment
        self._decision_engine = decision_engine
        self._action_validator = action_validator
        self._telemetry = telemetry
        self._max_steps = max_steps

    def run_episode(self) -> EpisodeResult:
        self._environment.reset()
        steps = 0
        total_reward = 0.0

        while not self._environment.is_finished():
            if self._max_steps is not None and steps >= self._max_steps:
                break

            observation = self._environment.observe()
            logger.debug("observation_created")
            decision = self._decision_engine.decide(observation)
            action = self._action_validator.validate(decision)
            result = self._environment.step(action)
            logger.debug(
                "action_executed action=%s reward=%.3f finished=%s",
                action.value,
                result.reward,
                result.finished,
            )
            self._telemetry.record(
                observation=observation,
                decision=decision,
                action=action,
                result=result,
            )

            steps += 1
            total_reward += result.reward

        return EpisodeResult(steps=steps, total_reward=total_reward)
