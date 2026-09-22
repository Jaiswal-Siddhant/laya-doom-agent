from __future__ import annotations

from laya_doom.observation.builder import serialize_observation
from laya_doom.observation.models import Observation


def test_identical_observations_serialize_identically() -> None:
    observation = Observation(
        health=85,
        armor=20,
        ammo=42,
        weapon="pistol",
        enemy_visible=True,
        enemy_distance=4.2,
        enemy_direction="left",
        episode_time=18.4,
        episode_finished=False,
    )

    assert serialize_observation(observation) == serialize_observation(observation.model_copy())
