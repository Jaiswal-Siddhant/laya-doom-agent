from __future__ import annotations

import math
from typing import Any

from laya_doom.observation.models import EnemyDirection, Observation


class ObservationBuilder:
    def __init__(self, vizdoom_module: Any | None = None) -> None:
        self._vzd = vizdoom_module

    def from_vizdoom(self, state: Any, game: Any) -> Observation:
        enemy = _nearest_enemy_label(getattr(state, "labels", []) or [])
        player_x = self._game_variable(game, "POSITION_X")
        player_y = self._game_variable(game, "POSITION_Y")
        player_angle = self._game_variable(game, "ANGLE")

        return Observation(
            health=int(self._game_variable(game, "HEALTH", default=0)),
            armor=int(self._game_variable(game, "ARMOR", default=0)),
            ammo=int(self._selected_weapon_ammo(game)),
            weapon=_weapon_name(int(self._game_variable(game, "SELECTED_WEAPON", default=0))),
            enemy_visible=enemy is not None,
            enemy_distance=_enemy_distance(enemy, player_x, player_y),
            enemy_direction=_enemy_direction(enemy, player_x, player_y, player_angle),
            episode_time=float(getattr(game, "get_episode_time")()),
            episode_finished=bool(getattr(game, "is_episode_finished")()),
        )

    def _selected_weapon_ammo(self, game: Any) -> float:
        selected_weapon_ammo = self._game_variable(game, "SELECTED_WEAPON_AMMO", default=None)
        if selected_weapon_ammo is not None:
            return selected_weapon_ammo
        return self._game_variable(game, "AMMO0", default=0)

    def _game_variable(self, game: Any, name: str, default: float | None = None) -> float | None:
        if self._vzd is None or not hasattr(self._vzd, "GameVariable"):
            return default
        enum_value = getattr(self._vzd.GameVariable, name, None)
        if enum_value is None:
            return default
        try:
            return float(game.get_game_variable(enum_value))
        except (AttributeError, RuntimeError, TypeError):
            return default


def serialize_observation(observation: Observation) -> str:
    enemy_distance = (
        "unknown" if observation.enemy_distance is None else f"{observation.enemy_distance:.2f}"
    )
    enemy_direction = observation.enemy_direction or "unknown"
    return "\n".join(
        [
            "DOOM STATE",
            "",
            "PLAYER",
            f"health: {observation.health}",
            f"armor: {observation.armor}",
            f"ammo: {observation.ammo}",
            f"weapon: {observation.weapon}",
            "",
            "COMBAT",
            f"enemy_visible: {str(observation.enemy_visible).lower()}",
            f"enemy_distance: {enemy_distance}",
            f"enemy_direction: {enemy_direction}",
            "",
            "EPISODE",
            f"elapsed_seconds: {observation.episode_time:.2f}",
            f"finished: {str(observation.episode_finished).lower()}",
        ]
    )


def _nearest_enemy_label(labels: list[Any]) -> Any | None:
    enemies = [label for label in labels if _is_enemy(label)]
    if not enemies:
        return None
    return min(enemies, key=lambda label: float(getattr(label, "object_position_x", 0.0)) ** 2)


def _is_enemy(label: Any) -> bool:
    name = str(getattr(label, "object_name", "")).lower()
    return any(token in name for token in ("zombieman", "shotgun", "imp", "demon", "cacodemon"))


def _enemy_distance(enemy: Any | None, player_x: float | None, player_y: float | None) -> float | None:
    if enemy is None or player_x is None or player_y is None:
        return None
    enemy_x = getattr(enemy, "object_position_x", None)
    enemy_y = getattr(enemy, "object_position_y", None)
    if enemy_x is None or enemy_y is None:
        return None
    return math.hypot(float(enemy_x) - player_x, float(enemy_y) - player_y)


def _enemy_direction(
    enemy: Any | None,
    player_x: float | None,
    player_y: float | None,
    player_angle: float | None,
) -> EnemyDirection | None:
    if enemy is None or player_x is None or player_y is None or player_angle is None:
        return None
    enemy_x = getattr(enemy, "object_position_x", None)
    enemy_y = getattr(enemy, "object_position_y", None)
    if enemy_x is None or enemy_y is None:
        return None

    angle_to_enemy = math.degrees(math.atan2(float(enemy_y) - player_y, float(enemy_x) - player_x))
    delta = ((angle_to_enemy - player_angle + 180) % 360) - 180
    if abs(delta) <= 15:
        return "center"
    return "left" if delta > 0 else "right"


def _weapon_name(weapon_id: int) -> str:
    return {
        0: "unknown",
        1: "fist",
        2: "pistol",
        3: "shotgun",
        4: "chaingun",
        5: "rocket_launcher",
        6: "plasma_rifle",
        7: "bfg",
        8: "chainsaw",
    }.get(weapon_id, f"weapon_{weapon_id}")
