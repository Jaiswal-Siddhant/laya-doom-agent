from __future__ import annotations

from laya_doom.actions.models import DoomAction
from laya_doom.actions.validator import ActionValidator
from laya_doom.decision.base import AgentDecision


def test_valid_laya_action_is_accepted() -> None:
    decision = AgentDecision(
        action=DoomAction.SHOOT,
        probability=0.81,
        latency_ms=13.2,
        raw_response=None,
    )

    assert ActionValidator().validate(decision) is DoomAction.SHOOT


def test_invalid_laya_action_falls_back_to_noop() -> None:
    assert ActionValidator().validate({"action": "open_console"}) is DoomAction.NOOP
