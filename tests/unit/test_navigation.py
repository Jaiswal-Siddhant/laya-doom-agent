from __future__ import annotations

from laya_doom.actions.models import DoomAction
from laya_doom.navigation.astar import AStarPlanner
from laya_doom.navigation.grid import LocalGridMap
from laya_doom.navigation.navigator import WALL_TURN_STEPS, FrontierNavigator
from laya_doom.observation.models import Observation


def test_astar_routes_around_a_blocked_cell() -> None:
    grid = LocalGridMap()
    for cell in ((0, 0), (0, 1), (1, 1), (2, 1), (2, 0)):
        grid.mark_traversable(cell)
    grid.mark_blocked((1, 0))

    path = AStarPlanner().find_path((0, 0), (2, 0), grid.known_neighbors)

    assert path == ((0, 0), (1, 1), (2, 0))


def test_navigator_marks_a_blocked_direction_then_replans() -> None:
    grid = LocalGridMap()
    navigator = FrontierNavigator(grid=grid)
    observation = _observation()

    assert navigator.decide(observation).action is DoomAction.MOVE_FORWARD
    assert navigator.recovery_decision(observation) is None
    assert navigator.recovery_decision(observation) is None
    decision = navigator.recovery_decision(observation)

    assert (1, 0) in grid.blocked
    assert decision.action is DoomAction.TURN_LEFT
    assert decision.reason == "wall_follow_turn"

    for _ in range(WALL_TURN_STEPS - 1):
        assert navigator.recovery_decision(observation).action is DoomAction.TURN_LEFT
    assert navigator.recovery_decision(observation).action is DoomAction.MOVE_FORWARD


def _observation() -> Observation:
    return Observation(
        health=100,
        armor=0,
        ammo=10,
        weapon="pistol",
        enemy_visible=False,
        player_x=0.0,
        player_y=0.0,
        player_angle=0.0,
        episode_time=1.0,
        episode_finished=False,
    )
