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
