from __future__ import annotations

import time
from typing import Any

from laya_doom.actions.models import DoomAction
from laya_doom.agent.prompts import LAYA_ACTION_QUESTION
from laya_doom.decision.base import AgentDecision
from laya_doom.observation.builder import serialize_observation
from laya_doom.observation.models import Observation


class LayaDecisionEngine:
    """Decision engine that contains all direct Laya-MLX API usage."""

    def __init__(self, model: str, dtype: str = "float16") -> None:
        try:
            import laya_mlx as laya
        except ModuleNotFoundError as exc:
            if exc.name == "mlx":
                msg = (
                    "Laya-MLX requires Apple's MLX runtime, which is available on "
                    "Apple Silicon/macOS. Run the real Laya decision engine on a "
                    "supported Mac, or run unit tests on this machine with "
                    "`python -m pytest tests/unit`."
                )
                raise LayaRuntimeUnavailable(msg) from exc
            if exc.name == "laya_mlx":
                msg = "laya-mlx is not installed. Install project dependencies first."
                raise LayaRuntimeUnavailable(msg) from exc
            raise

        self._agent = laya.load(model, dtype=dtype)
        self._questions = {
            "action": {
                "type": "choice",
                "instructions": LAYA_ACTION_QUESTION,
                "criteria": {
                    action.value: _action_description(action)
                    for action in DoomAction
                },
            }
        }

    def decide(self, observation: Observation) -> AgentDecision:
        state = serialize_observation(observation)
        started = time.perf_counter()
        result = self._agent.predict(state, self._questions)
        latency_ms = (time.perf_counter() - started) * 1000
        action, probability = _extract_choice(result, "action")
        return AgentDecision(
            action=action,
            probability=probability,
            latency_ms=latency_ms,
            raw_response=result if isinstance(result, dict) else {"response": result},
        )


def _action_description(action: DoomAction) -> str:
    return {
        DoomAction.MOVE_FORWARD: "advance toward the visible enemy or explore if no enemy is visible",
        DoomAction.TURN_LEFT: "rotate left to face an enemy or search the arena",
        DoomAction.TURN_RIGHT: "rotate right to face an enemy or search the arena",
        DoomAction.SHOOT: "fire the weapon when an enemy is visible and ammunition is available",
        DoomAction.NOOP: "do nothing when no useful or safe action is available",
    }[action]


def _extract_choice(result: Any, question_key: str) -> tuple[DoomAction, float | None]:
    if not isinstance(result, dict):
        return DoomAction.NOOP, None

    answer = result.get("answers", {}).get(question_key)
    if isinstance(answer, str):
        return _coerce_action(answer), None
    if not isinstance(answer, dict):
        return DoomAction.NOOP, None

    raw_value = (
        answer.get("value")
        or answer.get("answer")
        or answer.get("choice")
        or answer.get("label")
        or answer.get("selected")
    )
    probabilities = answer.get("probabilities") or answer.get("probs")
    probability = _extract_probability(raw_value, probabilities)

    action_block = answer.get("action")
    if probability is None and isinstance(action_block, dict):
        probability = _as_float(action_block.get("act_probability"))

    return _coerce_action(raw_value), probability


def _coerce_action(value: Any) -> DoomAction:
    try:
        return DoomAction(str(value))
    except ValueError:
        return DoomAction.NOOP


def _extract_probability(value: Any, probabilities: Any) -> float | None:
    if isinstance(probabilities, dict) and value is not None:
        return _as_float(probabilities.get(str(value)))
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


class LayaRuntimeUnavailable(RuntimeError):
    """Raised when the local Laya-MLX runtime cannot be imported on this platform."""
