from __future__ import annotations

from enum import Enum


class DoomAction(str, Enum):
    MOVE_FORWARD = "move_forward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    SHOOT = "shoot"
    NOOP = "noop"
