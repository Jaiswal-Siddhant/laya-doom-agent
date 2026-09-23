from __future__ import annotations

from laya_doom.actions.models import DoomAction


class ViZDoomActionMapper:
    """Maps domain actions to the button order supplied by a ViZDoom scenario."""

    _default_button_order = ("MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK")
    _action_buttons: dict[DoomAction, tuple[str, ...]] = {
        DoomAction.MOVE_FORWARD: ("MOVE_FORWARD",),
        DoomAction.MOVE_BACKWARD: ("MOVE_BACKWARD",),
        DoomAction.STRAFE_LEFT: ("STRAFE", "MOVE_LEFT"),
        DoomAction.STRAFE_RIGHT: ("STRAFE", "MOVE_RIGHT"),
        DoomAction.TURN_LEFT: ("TURN_LEFT",),
        DoomAction.TURN_RIGHT: ("TURN_RIGHT",),
        DoomAction.SHOOT: ("ATTACK",),
        DoomAction.USE: ("USE",),
        DoomAction.NOOP: (),
    }

    def __init__(self, button_order: tuple[str, ...] | None = None) -> None:
        self._button_order = button_order or self._default_button_order
        self._mapping = {
            action: [int(button in self._action_buttons[action]) for button in self._button_order]
            for action in DoomAction
        }

    def to_vizdoom(self, action: DoomAction) -> list[int]:
        return self._mapping[action].copy()

    @property
    def button_order(self) -> tuple[str, ...]:
        return self._button_order

    @property
    def available_actions(self) -> tuple[DoomAction, ...]:
        return tuple(
            action
            for action, buttons in self._action_buttons.items()
            if not buttons or all(button in self._button_order for button in buttons)
        )
