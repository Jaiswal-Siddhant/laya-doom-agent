from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TextIO

from laya_doom.actions.models import DoomAction
from laya_doom.decision.base import AgentDecision
from laya_doom.environment.base import StepResult
from laya_doom.observation.models import Observation


class TelemetryRecorder(Protocol):
    def record(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
    ) -> None: ...

    def close(self) -> None: ...


class JsonlTelemetryRecorder:
    def __init__(self, events_file: TextIO) -> None:
        self._events_file = events_file

    @classmethod
    def create(cls, telemetry_root: Path) -> JsonlTelemetryRecorder:
        run_dir = telemetry_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_dir.mkdir(parents=True, exist_ok=False)
        return cls((run_dir / "events.jsonl").open("a", encoding="utf-8"))

    def record(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
    ) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "health": observation.health,
            "armor": observation.armor,
            "ammo": observation.ammo,
            "enemy_visible": observation.enemy_visible,
            "visible_enemy_count": observation.visible_enemy_count,
            "target_name": observation.target_name,
            "enemy_in_crosshair": observation.enemy_in_crosshair,
            "enemy_distance": observation.enemy_distance,
            "enemy_direction": observation.enemy_direction,
            "action": action.value,
            "laya_action": _raw_response_value(decision, "laya_action"),
            "decision_source": _raw_response_value(decision, "source"),
            "navigation_reason": _raw_response_value(decision, "reason"),
            "probability": decision.probability,
            "latency_ms": decision.latency_ms,
            "reward": result.reward,
            "finished": result.finished,
        }
        self._events_file.write(json.dumps(payload, sort_keys=True) + "\n")
        self._events_file.flush()

    def close(self) -> None:
        self._events_file.close()


class NullTelemetryRecorder:
    def record(
        self,
        observation: Observation,
        decision: AgentDecision,
        action: DoomAction,
        result: StepResult,
    ) -> None:
        return None

    def close(self) -> None:
        return None


def _raw_response_value(decision: AgentDecision, key: str) -> str | None:
    if decision.raw_response is None:
        return None
    value = decision.raw_response.get(key)
    return value if isinstance(value, str) else None
