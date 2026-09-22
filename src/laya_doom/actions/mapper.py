from __future__ import annotations

from laya_doom.actions.models import DoomAction


class ViZDoomActionMapper:
    """Maps V1 domain actions to ViZDoom's binary button vector."""

    def __init__(self) -> None:
        self._mapping: dict[DoomAction, list[int]] = {
            DoomAction.MOVE_FORWARD: [1, 0, 0, 0],
            DoomAction.TURN_LEFT: [0, 1, 0, 0],
            DoomAction.TURN_RIGHT: [0, 0, 1, 0],
            DoomAction.SHOOT: [0, 0, 0, 1],
            DoomAction.NOOP: [0, 0, 0, 0],
        }

    def to_vizdoom(self, action: DoomAction) -> list[int]:
        return self._mapping[action].copy()

    @property
    def button_order(self) -> tuple[str, ...]:
        return ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK")
