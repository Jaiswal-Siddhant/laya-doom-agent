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
    enemy_distance: float | None = Field(default=None, ge=0.0)
    enemy_direction: EnemyDirection | None = None
    episode_time: float = Field(ge=0.0)
    episode_finished: bool
