from __future__ import annotations

from laya_doom.actions.mapper import ViZDoomActionMapper
from laya_doom.actions.models import DoomAction


def test_every_v1_action_maps_to_vizdoom_button_vector() -> None:
    mapper = ViZDoomActionMapper()

    assert mapper.to_vizdoom(DoomAction.MOVE_FORWARD) == [1, 0, 0, 0]
    assert mapper.to_vizdoom(DoomAction.TURN_LEFT) == [0, 1, 0, 0]
    assert mapper.to_vizdoom(DoomAction.TURN_RIGHT) == [0, 0, 1, 0]
    assert mapper.to_vizdoom(DoomAction.SHOOT) == [0, 0, 0, 1]
    assert mapper.to_vizdoom(DoomAction.NOOP) == [0, 0, 0, 0]


def test_full_level_buttons_are_mapped_by_name_not_position() -> None:
    mapper = ViZDoomActionMapper(
        ("ATTACK", "STRAFE", "USE", "MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD", "TURN_LEFT")
    )

    assert mapper.to_vizdoom(DoomAction.SHOOT) == [1, 0, 0, 0, 0, 0, 0]
    assert mapper.to_vizdoom(DoomAction.STRAFE_LEFT) == [0, 1, 0, 1, 0, 0, 0]
    assert mapper.to_vizdoom(DoomAction.USE) == [0, 0, 1, 0, 0, 0, 0]
    assert DoomAction.MOVE_BACKWARD not in mapper.available_actions
