from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EnemyDirection = Literal["left", "center", "right"]


class Observation(BaseModel):
    health: int = Field(ge=0)
    armor: int = Field(ge=0)
    ammo: int = Field(ge=0)
    weapon: str
    enemy_visible: bool
    visible_enemy_count: int = Field(default=0, ge=0)
    target_name: str | None = None
    enemy_in_crosshair: bool = False
    enemy_distance: float | None = Field(default=None, ge=0.0)
    enemy_direction: EnemyDirection | None = None
    player_x: float | None = None
    player_y: float | None = None
    player_angle: float | None = None
    episode_time: float = Field(ge=0.0)
    episode_finished: bool
