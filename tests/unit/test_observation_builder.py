from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from laya_doom.observation.builder import (
    AIM_ALIGNMENT_DEGREES,
    ObservationBuilder,
    _enemy_direction,
    _enemy_in_crosshair,
    _enemy_labels,
    _enemy_screen_direction,
    _is_enemy,
    _select_target,
)


class GameVariable:
    HEALTH = "HEALTH"
    ARMOR = "ARMOR"
    AMMO0 = "AMMO0"
    SELECTED_WEAPON = "SELECTED_WEAPON"
    SELECTED_WEAPON_AMMO = "SELECTED_WEAPON_AMMO"
    POSITION_X = "POSITION_X"
    POSITION_Y = "POSITION_Y"
    ANGLE = "ANGLE"


class FakeVzd:
    GameVariable = GameVariable


class FakeGame:
    def __init__(self, variables: dict[str, float]) -> None:
        self._variables = variables

    def get_game_variable(self, variable: str) -> float:
        return self._variables[variable]

    def get_episode_time(self) -> int:
        return 42

    def is_episode_finished(self) -> bool:
        return False

    def get_screen_width(self) -> int:
        return 320


def test_raw_vizdoom_state_becomes_observation() -> None:
    state = SimpleNamespace(
        labels=[
            SimpleNamespace(
                object_name="Zombieman",
                object_position_x=3.0,
                object_position_y=4.0,
                x=150,
                width=20,
            )
        ]
    )
    game = FakeGame(
        {
            "HEALTH": 85,
            "ARMOR": 20,
            "AMMO0": 5,
            "SELECTED_WEAPON": 2,
            "SELECTED_WEAPON_AMMO": 42,
            "POSITION_X": 0,
            "POSITION_Y": 0,
            "ANGLE": 53,
        }
    )

    observation = ObservationBuilder(FakeVzd).from_vizdoom(state, game)

    assert observation.health == 85
    assert observation.armor == 20
    assert observation.ammo == 42
    assert observation.weapon == "pistol"
    assert observation.enemy_visible is True
    assert observation.visible_enemy_count == 1
    assert observation.enemy_in_crosshair is True
    assert observation.enemy_distance == 5.0
    assert observation.enemy_direction == "center"
    assert observation.episode_time == 42.0
    assert observation.episode_finished is False


def test_missing_enemy_keeps_enemy_values_unknown() -> None:
    state = SimpleNamespace(labels=[])
    game: Any = FakeGame(
        {
            "HEALTH": 100,
            "ARMOR": 0,
            "AMMO0": 10,
            "SELECTED_WEAPON": 2,
            "SELECTED_WEAPON_AMMO": 10,
            "POSITION_X": 0,
            "POSITION_Y": 0,
            "ANGLE": 0,
        }
    )

    observation = ObservationBuilder(FakeVzd).from_vizdoom(state, game)

    assert observation.enemy_visible is False
    assert observation.enemy_distance is None
    assert observation.enemy_direction is None


@pytest.mark.parametrize(
    ("target_angle", "expected"),
    [
        (AIM_ALIGNMENT_DEGREES, "center"),
        (AIM_ALIGNMENT_DEGREES + 1, "left"),
        (-(AIM_ALIGNMENT_DEGREES + 1), "right"),
    ],
)
def test_enemy_direction_requires_precise_alignment(target_angle: int, expected: str) -> None:
    enemy = SimpleNamespace(
        object_position_x=math.cos(math.radians(target_angle)),
        object_position_y=math.sin(math.radians(target_angle)),
    )

    assert _enemy_direction(enemy, 0.0, 0.0, 0.0) == expected


def test_enemy_in_crosshair_requires_label_to_overlap_screen_center() -> None:
    assert _enemy_in_crosshair(SimpleNamespace(x=150, width=20), 320) is True
    assert _enemy_in_crosshair(SimpleNamespace(x=120, width=20), 320) is False


def test_enemy_screen_direction_tracks_the_crosshair_position() -> None:
    assert _enemy_screen_direction(SimpleNamespace(x=100, width=20), 320) == "left"
    assert _enemy_screen_direction(SimpleNamespace(x=150, width=20), 320) == "center"
    assert _enemy_screen_direction(SimpleNamespace(x=200, width=20), 320) == "right"


def test_target_selection_prioritizes_crosshair_then_distance() -> None:
    nearest_off_crosshair = SimpleNamespace(
        object_position_x=10,
        object_position_y=0,
        x=0,
        width=10,
    )
    centered_enemy = SimpleNamespace(
        object_position_x=100,
        object_position_y=0,
        x=150,
        width=20,
    )

    target = _select_target([nearest_off_crosshair, centered_enemy], 0.0, 0.0, 320)

    assert target is centered_enemy


def test_target_selection_uses_true_two_dimensional_distance_without_crosshair_target() -> None:
    farther_enemy = SimpleNamespace(object_position_x=100, object_position_y=0, x=0, width=10)
    nearer_enemy = SimpleNamespace(object_position_x=3, object_position_y=4, x=250, width=10)

    target = _select_target([farther_enemy, nearer_enemy], 0.0, 0.0, 320)

    assert target is nearer_enemy


@pytest.mark.parametrize("name", ["ChaingunGuy", "FormerHuman", "MarineChainsawVzd", "Zombieman"])
def test_humanoid_enemy_names_are_recognized(name: str) -> None:
    assert _is_enemy(SimpleNamespace(object_name=name)) is True


def test_self_label_is_not_selected_as_an_enemy() -> None:
    self_label = SimpleNamespace(object_name="DoomPlayer", object_position_x=0, object_position_y=0)
    enemy = SimpleNamespace(
        object_name="MarineChainsawVzd", object_position_x=100, object_position_y=0
    )

    enemies = _enemy_labels([self_label, enemy], 0.0, 0.0)

    assert enemies == [enemy]
