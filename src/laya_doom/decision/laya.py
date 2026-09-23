from __future__ import annotations

import logging
import time
from typing import Any

from laya_doom.actions.models import DoomAction
from laya_doom.agent.prompts import LAYA_ACTION_QUESTION
from laya_doom.decision.base import AgentDecision
from laya_doom.navigation.navigator import FrontierNavigator, NavigationDecision
from laya_doom.observation.builder import serialize_observation
from laya_doom.observation.models import Observation

logger = logging.getLogger(__name__)


class LayaDecisionEngine:
    """Decision engine that contains all direct Laya-MLX API usage."""

    def __init__(
        self, model: str, dtype: str = "float16", actions: tuple[DoomAction, ...] | None = None
    ) -> None:
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
        self._available_actions = actions or tuple(DoomAction)
        self._navigator = FrontierNavigator()

    def decide(self, observation: Observation) -> AgentDecision:
        navigation = self._navigation_decision(observation)
        if navigation is not None:
            return self._navigation_agent_decision(observation, navigation)

        state = serialize_observation(observation)
        started = time.perf_counter()
        result = self._agent.predict(state, self._questions_for(observation))
        latency_ms = (time.perf_counter() - started) * 1000
        laya_action, probability = _extract_choice(result, "action")
        logger.info(
            "decision laya_action=%s action=%s source=%s enemy_visible=%s "
            "enemy_in_crosshair=%s enemy_direction=%s enemy_distance=%s ammo=%s "
            "choices=%s probability=%s latency_ms=%.1f",
            laya_action.value,
            laya_action.value,
            "laya",
            observation.enemy_visible,
            observation.enemy_in_crosshair,
            observation.enemy_direction,
            observation.enemy_distance,
            observation.ammo,
            ",".join(self._action_choices(observation)),
            probability,
            latency_ms,
        )
        return AgentDecision(
            action=laya_action,
            probability=probability,
            latency_ms=latency_ms,
            raw_response={
                "laya_action": laya_action.value,
                "source": "laya",
                "choices": self._action_choices(observation),
                "response": result,
            },
        )

    def _navigation_decision(self, observation: Observation) -> NavigationDecision | None:
        if DoomAction.MOVE_FORWARD not in self._available_actions:
            return None
        if not observation.enemy_visible:
            return self._navigator.decide(observation)
        if not observation.enemy_in_crosshair:
            return self._navigator.recovery_decision(observation)
        return None

    def record_action(self, action: DoomAction) -> None:
        self._navigator.record_action(action)

    def _navigation_agent_decision(
        self, observation: Observation, navigation: NavigationDecision
    ) -> AgentDecision:
        logger.info(
            "decision action=%s source=navigator enemy_visible=%s waypoint=%s reason=%s",
            navigation.action.value,
            observation.enemy_visible,
            navigation.waypoint,
            navigation.reason,
        )
        return AgentDecision(
            action=navigation.action,
            probability=None,
            latency_ms=0.0,
            raw_response={
                "source": "navigator",
                "waypoint": navigation.waypoint,
                "reason": navigation.reason,
            },
        )

    def _questions_for(self, observation: Observation) -> dict[str, Any]:
        actions = self._actions_for(observation)
        return {
            "action": {
                "type": "choice",
                "instructions": LAYA_ACTION_QUESTION,
                "criteria": {action.value: _action_description(action) for action in actions},
            }
        }

    def _action_choices(self, observation: Observation) -> tuple[str, ...]:
        return tuple(action.value for action in self._actions_for(observation))

    def _actions_for(self, observation: Observation) -> tuple[DoomAction, ...]:
        # A living player should keep exploring or engaging. NOOP remains available
        # as the validator's fallback, but is not a productive Laya choice.
        actions = tuple(
            action for action in self._available_actions if action is not DoomAction.NOOP
        )
        if observation.enemy_visible and observation.enemy_direction == "left":
            return (DoomAction.TURN_LEFT,)
        if observation.enemy_visible and observation.enemy_direction == "right":
            return (DoomAction.TURN_RIGHT,)
        if observation.enemy_visible and observation.enemy_in_crosshair and observation.ammo > 0:
            actions = tuple(
                action
                for action in actions
                if action
                in (
                    DoomAction.SHOOT,
                    DoomAction.STRAFE_LEFT,
                    DoomAction.STRAFE_RIGHT,
                    DoomAction.MOVE_BACKWARD,
                )
            )
        else:
            actions = tuple(action for action in actions if action is not DoomAction.SHOOT)

        return actions


def _action_description(action: DoomAction) -> str:
    return {
        DoomAction.MOVE_FORWARD: (
            "advance toward the visible enemy or explore if no enemy is visible"
        ),
        DoomAction.MOVE_BACKWARD: "retreat from a threat or back away from an obstacle",
        DoomAction.STRAFE_LEFT: "move left while keeping the current aim direction",
        DoomAction.STRAFE_RIGHT: "move right while keeping the current aim direction",
        DoomAction.TURN_LEFT: "rotate left to face an enemy or search the arena",
        DoomAction.TURN_RIGHT: "rotate right to face an enemy or search the arena",
        DoomAction.SHOOT: "fire the weapon when an enemy is visible and ammunition is available",
        DoomAction.USE: "open a nearby door, activate a switch, or use an exit",
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
