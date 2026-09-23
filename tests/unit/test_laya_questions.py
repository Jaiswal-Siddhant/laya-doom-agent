from __future__ import annotations

from laya_doom.actions.models import DoomAction
from laya_doom.decision.laya import LayaDecisionEngine
from laya_doom.observation.models import Observation


def test_laya_can_only_choose_shoot_when_enemy_overlaps_crosshair() -> None:
    engine = LayaDecisionEngine.__new__(LayaDecisionEngine)
    engine._available_actions = tuple(DoomAction)
    observation = Observation(
        health=100,
        armor=0,
        ammo=10,
        weapon="pistol",
        enemy_visible=True,
        enemy_in_crosshair=False,
        enemy_distance=25.0,
        enemy_direction="left",
        episode_time=1.0,
        episode_finished=False,
    )

    criteria = engine._questions_for(observation)["action"]["criteria"]

    assert DoomAction.SHOOT.value not in criteria
    assert DoomAction.NOOP.value not in criteria
    aimed_observation = observation.model_copy(
        update={"enemy_in_crosshair": True, "enemy_direction": "center"}
    )
    criteria = engine._questions_for(aimed_observation)["action"]["criteria"]
    assert DoomAction.SHOOT.value in criteria


def test_laya_must_turn_toward_a_visible_off_crosshair_enemy() -> None:
    engine = LayaDecisionEngine.__new__(LayaDecisionEngine)
    engine._available_actions = tuple(DoomAction)
    observation = Observation(
        health=100,
        armor=0,
        ammo=10,
        weapon="pistol",
        enemy_visible=True,
        enemy_in_crosshair=False,
        enemy_distance=25.0,
        enemy_direction="right",
        episode_time=1.0,
        episode_finished=False,
    )

    assert engine._action_choices(observation) == (DoomAction.TURN_RIGHT.value,)


def test_static_combat_scenario_does_not_enable_navigation() -> None:
    engine = LayaDecisionEngine.__new__(LayaDecisionEngine)
    engine._available_actions = (DoomAction.TURN_LEFT, DoomAction.TURN_RIGHT, DoomAction.SHOOT)
    engine._navigator = object()

    assert engine._navigation_decision(_exploration_observation()) is None


def _exploration_observation() -> Observation:
    return Observation(
        health=100,
        armor=0,
        ammo=10,
        weapon="pistol",
        enemy_visible=False,
        episode_time=1.0,
        episode_finished=False,
    )
