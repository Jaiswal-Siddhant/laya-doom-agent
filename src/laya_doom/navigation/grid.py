from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

GridCell = tuple[int, int]


@dataclass
class LocalGridMap:
    """A sparse occupancy map learned from successful and blocked movement."""

    traversable: set[GridCell] = field(default_factory=set)
    blocked: set[GridCell] = field(default_factory=set)

    def mark_traversable(self, cell: GridCell) -> None:
        self.blocked.discard(cell)
        self.traversable.add(cell)

    def mark_blocked(self, cell: GridCell) -> None:
        if cell not in self.traversable:
            self.blocked.add(cell)

    def known_neighbors(self, cell: GridCell) -> tuple[GridCell, ...]:
        return tuple(neighbor for neighbor in _neighbors(cell) if neighbor in self.traversable)

    def unknown_neighbors(self, cell: GridCell) -> tuple[GridCell, ...]:
        return tuple(
            neighbor
            for neighbor in _neighbors(cell)
            if neighbor not in self.traversable and neighbor not in self.blocked
        )

    def frontier_cells(self) -> tuple[GridCell, ...]:
        return tuple(cell for cell in self.traversable if self.unknown_neighbors(cell))


def neighbors(cell: GridCell) -> Iterable[GridCell]:
    return _neighbors(cell)


def _neighbors(cell: GridCell) -> tuple[GridCell, ...]:
    x, y = cell
    return (
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1),
        (x - 1, y + 1),
        (x - 1, y),
        (x - 1, y - 1),
        (x, y - 1),
        (x + 1, y - 1),
    )
