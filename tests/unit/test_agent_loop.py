from __future__ import annotations

from laya_doom.actions.models import DoomAction
from laya_doom.actions.validator import ActionValidator
from laya_doom.agent.agent import DoomAgent
from laya_doom.decision.base import AgentDecision
from laya_doom.environment.base import StepResult
from laya_doom.observation.models import Observation
from laya_doom.telemetry.events import NullTelemetryRecorder


class FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._finished = False

    def reset(self) -> None:
        self.calls.append("reset")

    def observe(self) -> Observation:
        self.calls.append("observe")
        return Observation(
            health=100,
            armor=0,
            ammo=10,
            weapon="pistol",
            enemy_visible=True,
            enemy_distance=5.0,
            enemy_direction="center",
            episode_time=1.0,
            episode_finished=False,
        )

    def step(self, action: DoomAction) -> StepResult:
        self.calls.append(f"step:{action.value}")
        self._finished = True
        return StepResult(reward=1.0, finished=True)

    def is_finished(self) -> bool:
        self.calls.append("is_finished")
        return self._finished

    def close(self) -> None:
        self.calls.append("close")


class FakeDecisionEngine:
    def decide(self, observation: Observation) -> AgentDecision:
        assert observation.enemy_visible is True
        return AgentDecision(
            action=DoomAction.SHOOT,
            probability=0.9,
            latency_ms=1.0,
            raw_response=None,
        )


def test_agent_loop_observes_decides_validates_and_steps() -> None:
    environment = FakeEnvironment()
    agent = DoomAgent(
        environment=environment,
        decision_engine=FakeDecisionEngine(),
        action_validator=ActionValidator(),
        telemetry=NullTelemetryRecorder(),
    )

    result = agent.run_episode()

    assert result.steps == 1
    assert result.total_reward == 1.0
    assert environment.calls == ["reset", "is_finished", "observe", "step:shoot", "is_finished"]
