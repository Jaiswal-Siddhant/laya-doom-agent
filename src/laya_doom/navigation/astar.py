from __future__ import annotations

import heapq
from collections.abc import Callable

from laya_doom.navigation.grid import GridCell


class AStarPlanner:
    """Shortest-path planner over a caller-supplied local traversability graph."""

    def find_path(
        self,
        start: GridCell,
        goal: GridCell,
        neighbors: Callable[[GridCell], tuple[GridCell, ...]],
    ) -> tuple[GridCell, ...] | None:
        queue: list[tuple[int, int, GridCell]] = [(0, 0, start)]
        came_from: dict[GridCell, GridCell | None] = {start: None}
        cost: dict[GridCell, int] = {start: 0}
        sequence = 0

        while queue:
            _, _, current = heapq.heappop(queue)
            if current == goal:
                return _reconstruct_path(came_from, goal)

            for neighbor in neighbors(current):
                next_cost = cost[current] + 1
                if next_cost >= cost.get(neighbor, float("inf")):
                    continue
                cost[neighbor] = next_cost
                came_from[neighbor] = current
                sequence += 1
                priority = next_cost + _manhattan_distance(neighbor, goal)
                heapq.heappush(queue, (priority, sequence, neighbor))
        return None


def _reconstruct_path(
    came_from: dict[GridCell, GridCell | None], goal: GridCell
) -> tuple[GridCell, ...]:
    path: list[GridCell] = []
    current: GridCell | None = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    return tuple(reversed(path))


def _manhattan_distance(first: GridCell, second: GridCell) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])
