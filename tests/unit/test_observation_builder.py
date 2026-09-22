from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from laya_doom.observation.builder import ObservationBuilder


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


def test_raw_vizdoom_state_becomes_observation() -> None:
    state = SimpleNamespace(
        labels=[
            SimpleNamespace(
                object_name="Zombieman",
                object_position_x=3.0,
                object_position_y=4.0,
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
            "ANGLE": 45,
        }
    )

    observation = ObservationBuilder(FakeVzd).from_vizdoom(state, game)

    assert observation.health == 85
    assert observation.armor == 20
    assert observation.ammo == 42
    assert observation.weapon == "pistol"
    assert observation.enemy_visible is True
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
