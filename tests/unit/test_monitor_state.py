from __future__ import annotations

from laya_doom.actions.models import DoomAction
from laya_doom.decision.base import AgentDecision
from laya_doom.environment.base import StepResult
from laya_doom.observation.models import Observation
from laya_doom.ui.window import MonitorState


def test_monitor_state_exposes_latest_decision_details() -> None:
    monitor = MonitorState("aac6fef/laya-mlx")
    observation = Observation(
        health=100,
        armor=0,
        ammo=10,
        weapon="pistol",
        enemy_visible=True,
        visible_enemy_count=2,
        target_name="MarineChainsawVzd",
        enemy_in_crosshair=True,
        episode_time=1.0,
        episode_finished=False,
    )
    decision = AgentDecision(
        action=DoomAction.SHOOT,
        probability=0.91,
        latency_ms=42.5,
        raw_response={"laya_action": "shoot", "source": "laya"},
    )

    monitor.publish(observation, decision, DoomAction.SHOOT, StepResult(reward=1, finished=False))

    snapshot = monitor.snapshot()
    assert snapshot["model"] == "aac6fef/laya-mlx"
    assert snapshot["latest"] == {
        "timestamp": snapshot["latest"]["timestamp"],
        "laya_action": "shoot",
        "action": "shoot",
        "latency_ms": 42.5,
        "probability": 0.91,
        "target_name": "MarineChainsawVzd",
        "enemy_count": 2,
        "enemy_in_crosshair": True,
        "ammo": 10,
        "reward": 1.0,
        "source": "laya",
    }
