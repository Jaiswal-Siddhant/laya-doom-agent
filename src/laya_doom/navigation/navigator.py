from __future__ import annotations

import math
from dataclasses import dataclass

from laya_doom.actions.models import DoomAction
from laya_doom.navigation.astar import AStarPlanner
from laya_doom.navigation.grid import GridCell, LocalGridMap
from laya_doom.observation.models import Observation

CELL_SIZE = 32.0
TURN_ALIGNMENT_DEGREES = 8.0
STUCK_STEP_LIMIT = 3
WALL_TURN_STEPS = 18
WALL_ADVANCE_STEPS = 8


@dataclass(frozen=True)
class NavigationDecision:
    action: DoomAction
    reason: str
    waypoint: GridCell


class FrontierNavigator:
    """Explores reachable frontiers and replans around locally discovered obstacles."""

    def __init__(
        self, grid: LocalGridMap | None = None, planner: AStarPlanner | None = None
    ) -> None:
        self._grid = grid or LocalGridMap()
        self._planner = planner or AStarPlanner()
        self._last_cell: GridCell | None = None
        self._last_position: tuple[float, float] | None = None
        self._last_action: DoomAction | None = None
        self._last_heading = 0.0
        self._stuck_steps = 0
        self._last_replan = False
        self._wall_follow_action: DoomAction | None = None
        self._wall_follow_steps = 0
        self._wall_follow_phase: str | None = None
        self._next_wall_turn = DoomAction.TURN_LEFT

    def decide(self, observation: Observation) -> NavigationDecision | None:
        current = self._update(observation)
        if current is None or observation.player_angle is None:
            return None
        wall_follow = self._wall_follow_decision(current)
        if wall_follow is not None:
            self._last_action = wall_follow.action
            return wall_follow
        waypoint, reason = self._next_waypoint(current)
        action = _action_toward(current, waypoint, observation.player_angle)
        self._last_action = action
        return NavigationDecision(action=action, reason=reason, waypoint=waypoint)

    def recovery_decision(self, observation: Observation) -> NavigationDecision | None:
        current = self._update(observation)
        if current is None or observation.player_angle is None:
            return None
        wall_follow = self._wall_follow_decision(current)
        if wall_follow is not None:
            self._last_action = wall_follow.action
            return wall_follow
        if not self._last_replan:
            return None
        waypoint, _ = self._next_waypoint(current)
        action = _action_toward(current, waypoint, observation.player_angle)
        self._last_action = action
        return NavigationDecision(action=action, reason="blocked_replan", waypoint=waypoint)

    def record_action(self, action: DoomAction) -> None:
        self._last_action = action

    def _update(self, observation: Observation) -> GridCell | None:
        current = _cell_for(observation)
        position = _position_for(observation)
        if current is None or position is None or observation.player_angle is None:
            return None
        self._last_replan = False
        self._advance_wall_follow()
        self._learn_from_last_move(current, position)
        self._grid.mark_traversable(current)
        self._last_cell = current
        self._last_position = position
        self._last_heading = observation.player_angle
        return current

    def _learn_from_last_move(self, current: GridCell, position: tuple[float, float]) -> None:
        if self._last_cell is None or self._last_action is not DoomAction.MOVE_FORWARD:
            self._stuck_steps = 0
            return
        if self._last_position is not None and _distance(position, self._last_position) >= 1.0:
            self._stuck_steps = 0
            return
        self._stuck_steps += 1
        if self._stuck_steps >= STUCK_STEP_LIMIT:
            self._stuck_steps = 0
            if self._wall_follow_action is None:
                self._grid.mark_blocked(_forward_cell(current, self._last_heading))
                self._last_replan = True
                self._start_wall_follow()

    def _start_wall_follow(self) -> None:
        self._wall_follow_action = self._next_wall_turn
        self._wall_follow_steps = WALL_TURN_STEPS
        self._wall_follow_phase = "turn"
        self._next_wall_turn = (
            DoomAction.TURN_RIGHT
            if self._next_wall_turn is DoomAction.TURN_LEFT
            else DoomAction.TURN_LEFT
        )

    def _advance_wall_follow(self) -> None:
        if self._wall_follow_action is None or self._last_action is not self._wall_follow_action:
            return
        self._wall_follow_steps -= 1
        if self._wall_follow_steps > 0:
            return
        if self._wall_follow_phase == "turn":
            self._wall_follow_action = DoomAction.MOVE_FORWARD
            self._wall_follow_steps = WALL_ADVANCE_STEPS
            self._wall_follow_phase = "advance"
            return
        self._wall_follow_action = None
        self._wall_follow_phase = None

    def _wall_follow_decision(self, current: GridCell) -> NavigationDecision | None:
        if self._wall_follow_action is None:
            return None
        reason = f"wall_follow_{self._wall_follow_phase}"
        return NavigationDecision(action=self._wall_follow_action, reason=reason, waypoint=current)

    def _next_waypoint(self, current: GridCell) -> tuple[GridCell, str]:
        frontiers = self._grid.frontier_cells()
        paths = [
            path
            for frontier in frontiers
            if (path := self._planner.find_path(current, frontier, self._grid.known_neighbors))
        ]
        if paths:
            path = min(paths, key=len)
            frontier = path[-1]
            unknown = self._grid.unknown_neighbors(frontier)
            if len(path) > 1:
                return path[1], "path_to_frontier"
            return unknown[0], "explore_frontier"
        return (current[0] + 1, current[1]), "bootstrap_exploration"


def _cell_for(observation: Observation) -> GridCell | None:
    if observation.player_x is None or observation.player_y is None:
        return None
    return round(observation.player_x / CELL_SIZE), round(observation.player_y / CELL_SIZE)


def _position_for(observation: Observation) -> tuple[float, float] | None:
    if observation.player_x is None or observation.player_y is None:
        return None
    return observation.player_x, observation.player_y


def _action_toward(current: GridCell, waypoint: GridCell, player_angle: float) -> DoomAction:
    target_angle = math.degrees(math.atan2(waypoint[1] - current[1], waypoint[0] - current[0]))
    delta = ((target_angle - player_angle + 180) % 360) - 180
    if abs(delta) <= TURN_ALIGNMENT_DEGREES:
        return DoomAction.MOVE_FORWARD
    return DoomAction.TURN_LEFT if delta > 0 else DoomAction.TURN_RIGHT


def _forward_cell(cell: GridCell, player_angle: float) -> GridCell:
    radians = math.radians(player_angle)
    dx = round(math.cos(radians))
    dy = round(math.sin(radians))
    return cell[0] + dx, cell[1] + dy


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])
