from __future__ import annotations

import math
from typing import Any

from laya_doom.observation.models import EnemyDirection, Observation

AIM_ALIGNMENT_DEGREES = 5
SELF_DISTANCE_THRESHOLD = 0.01


class ObservationBuilder:
    def __init__(self, vizdoom_module: Any | None = None) -> None:
        self._vzd = vizdoom_module

    def from_vizdoom(self, state: Any, game: Any) -> Observation:
        player_x = self._game_variable(game, "POSITION_X")
        player_y = self._game_variable(game, "POSITION_Y")
        player_angle = self._game_variable(game, "ANGLE")
        screen_width = self._screen_width(game)
        enemies = _enemy_labels(getattr(state, "labels", []) or [], player_x, player_y)
        enemy = _select_target(enemies, player_x, player_y, screen_width)

        return Observation(
            health=int(self._game_variable(game, "HEALTH", default=0) or 0),
            armor=int(self._game_variable(game, "ARMOR", default=0) or 0),
            ammo=int(self._selected_weapon_ammo(game)),
            weapon=_weapon_name(int(self._game_variable(game, "SELECTED_WEAPON", default=0) or 0)),
            enemy_visible=enemy is not None,
            visible_enemy_count=len(enemies),
            target_name=_label_name(enemy),
            enemy_in_crosshair=_enemy_in_crosshair(enemy, screen_width),
            enemy_distance=_enemy_distance(enemy, player_x, player_y),
            enemy_direction=_enemy_direction(enemy, player_x, player_y, player_angle, screen_width),
            player_x=player_x,
            player_y=player_y,
            player_angle=player_angle,
            episode_time=float(game.get_episode_time()),
            episode_finished=bool(game.is_episode_finished()),
        )

    def _selected_weapon_ammo(self, game: Any) -> float:
        selected_weapon_ammo = self._game_variable(game, "SELECTED_WEAPON_AMMO", default=0)
        ammo_pools = [
            self._game_variable(game, f"AMMO{index}", default=0) or 0 for index in range(8)
        ]
        return max(selected_weapon_ammo or 0, *ammo_pools)

    def _game_variable(self, game: Any, name: str, default: float | None = None) -> float | None:
        if self._vzd is None or not hasattr(self._vzd, "GameVariable"):
            return default
        enum_value = getattr(self._vzd.GameVariable, name, None)
        if enum_value is None:
            return default
        try:
            return float(game.get_game_variable(enum_value))
        except (AttributeError, KeyError, RuntimeError, TypeError):
            return default

    def _screen_width(self, game: Any) -> int | None:
        try:
            return int(game.get_screen_width())
        except (AttributeError, RuntimeError, TypeError):
            return None


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
            f"visible_enemy_count: {observation.visible_enemy_count}",
            f"target_name: {observation.target_name or 'unknown'}",
            f"enemy_in_crosshair: {str(observation.enemy_in_crosshair).lower()}",
            f"enemy_distance: {enemy_distance}",
            f"enemy_direction: {enemy_direction}",
            "",
            "EPISODE",
            f"elapsed_seconds: {observation.episode_time:.2f}",
            f"finished: {str(observation.episode_finished).lower()}",
            "",
            "POSITION",
            f"x: {_format_position(observation.player_x)}",
            f"y: {_format_position(observation.player_y)}",
            f"angle: {_format_position(observation.player_angle)}",
        ]
    )


def _enemy_labels(labels: list[Any], player_x: float | None, player_y: float | None) -> list[Any]:
    return [
        label
        for label in labels
        if _is_enemy(label) and not _is_self_label(label, player_x, player_y)
    ]


def _select_target(
    enemies: list[Any], player_x: float | None, player_y: float | None, screen_width: int | None
) -> Any | None:
    if not enemies:
        return None
    return min(
        enemies,
        key=lambda enemy: (
            not _enemy_in_crosshair(enemy, screen_width),
            _enemy_distance(enemy, player_x, player_y) or float("inf"),
        ),
    )


def _is_enemy(label: Any) -> bool:
    name = _label_name(label)
    return name is not None and any(token in name.lower() for token in _ENEMY_NAME_TOKENS)


_ENEMY_NAME_TOKENS = (
    "arachnotron",
    "archvile",
    "baron",
    "cacodemon",
    "chaingunguy",
    "cyberdemon",
    "demon",
    "formerhuman",
    "hellknight",
    "imp",
    "lostsoul",
    "mancubus",
    "marine",
    "pain",
    "revenant",
    "shotgunguy",
    "spectre",
    "spider",
    "zombieman",
)


def _label_name(label: Any | None) -> str | None:
    if label is None:
        return None
    name = str(getattr(label, "object_name", "")).strip()
    return name or None


def _is_self_label(label: Any, player_x: float | None, player_y: float | None) -> bool:
    name = _label_name(label)
    if name is not None and name.lower() == "doomplayer":
        return True
    distance = _enemy_distance(label, player_x, player_y)
    return distance is not None and distance < SELF_DISTANCE_THRESHOLD


def _enemy_in_crosshair(enemy: Any | None, screen_width: int | None) -> bool:
    if enemy is None or screen_width is None:
        return False
    left = getattr(enemy, "x", None)
    width = getattr(enemy, "width", None)
    if left is None or width is None:
        return False
    crosshair_x = screen_width / 2
    return float(left) <= crosshair_x <= float(left) + float(width)


def _enemy_distance(
    enemy: Any | None, player_x: float | None, player_y: float | None
) -> float | None:
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
    screen_width: int | None = None,
) -> EnemyDirection | None:
    screen_direction = _enemy_screen_direction(enemy, screen_width)
    if screen_direction is not None:
        return screen_direction
    if enemy is None or player_x is None or player_y is None or player_angle is None:
        return None
    enemy_x = getattr(enemy, "object_position_x", None)
    enemy_y = getattr(enemy, "object_position_y", None)
    if enemy_x is None or enemy_y is None:
        return None

    angle_to_enemy = math.degrees(math.atan2(float(enemy_y) - player_y, float(enemy_x) - player_x))
    delta = ((angle_to_enemy - player_angle + 180) % 360) - 180
    if abs(delta) <= AIM_ALIGNMENT_DEGREES:
        return "center"
    return "left" if delta > 0 else "right"


def _enemy_screen_direction(enemy: Any | None, screen_width: int | None) -> EnemyDirection | None:
    if enemy is None or screen_width is None:
        return None
    left = getattr(enemy, "x", None)
    width = getattr(enemy, "width", None)
    if left is None or width is None:
        return None
    crosshair_x = screen_width / 2
    if float(left) <= crosshair_x <= float(left) + float(width):
        return "center"
    enemy_center = float(left) + float(width) / 2
    return "left" if enemy_center < crosshair_x else "right"


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


def _format_position(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"
