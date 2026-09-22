from __future__ import annotations

import logging
from typing import Any

from laya_doom.actions.models import DoomAction
from laya_doom.decision.base import AgentDecision

logger = logging.getLogger(__name__)


class ActionValidator:
    def validate(self, decision: AgentDecision | str | dict[str, Any]) -> DoomAction:
        value = self._extract_action(decision)
        try:
            return DoomAction(value)
        except ValueError:
            logger.warning("invalid_action fallback=%s raw_action=%r", DoomAction.NOOP.value, value)
            return DoomAction.NOOP

    def _extract_action(self, decision: AgentDecision | str | dict[str, Any]) -> Any:
        if isinstance(decision, AgentDecision):
            return decision.action.value
        if isinstance(decision, str):
            return decision
        return decision.get("action")
