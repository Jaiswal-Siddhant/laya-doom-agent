from __future__ import annotations

from enum import StrEnum


class DoomAction(StrEnum):
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SHOOT = "shoot"
    USE = "use"
    NOOP = "noop"
