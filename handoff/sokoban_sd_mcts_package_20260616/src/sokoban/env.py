"""Sokoban parsing and transition utilities for Boxoban-style levels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np


Cell = Tuple[int, int]
DIRECTIONS: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTION_NAMES: Tuple[str, ...] = ("up", "down", "left", "right")


def add_cells(a: Cell, b: Cell) -> Cell:
    return a[0] + b[0], a[1] + b[1]


@dataclass(frozen=True)
class SokobanState:
    """Dynamic Sokoban state with box identities quotiented out by sorting."""

    player: Cell
    boxes: Tuple[Cell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "boxes", tuple(sorted(self.boxes)))


@dataclass(frozen=True)
class SokobanLevel:
    """Static Sokoban layout plus initial dynamic state."""

    level_id: str
    grid: Tuple[str, ...]
    walls: frozenset[Cell]
    goals: frozenset[Cell]
    initial_state: SokobanState

    @property
    def height(self) -> int:
        return len(self.grid)

    @property
    def width(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    @property
    def n_boxes(self) -> int:
        return len(self.initial_state.boxes)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_wall(self, cell: Cell) -> bool:
        return (not self.in_bounds(cell)) or cell in self.walls

    def is_floor(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self.walls

    def is_free(self, cell: Cell, boxes: Iterable[Cell]) -> bool:
        return self.is_floor(cell) and cell not in set(boxes)


def parse_sokoban_grid(lines: Sequence[str], level_id: str = "") -> SokobanLevel:
    """Parse one Sokoban ASCII grid."""

    if not lines:
        raise ValueError("Cannot parse an empty Sokoban grid.")
    width = max(len(line.rstrip("\n")) for line in lines)
    norm_lines = tuple(line.rstrip("\n").ljust(width) for line in lines)

    walls: set[Cell] = set()
    goals: set[Cell] = set()
    boxes: list[Cell] = []
    player: Optional[Cell] = None
    for r, line in enumerate(norm_lines):
        for c, ch in enumerate(line):
            cell = (r, c)
            if ch == "#":
                walls.add(cell)
            elif ch in ".+*":
                goals.add(cell)
            if ch in "$*":
                boxes.append(cell)
            elif ch in "@+":
                player = cell

    if player is None:
        raise ValueError(f"Level {level_id!r} has no player.")
    if not boxes:
        raise ValueError(f"Level {level_id!r} has no boxes.")
    if len(boxes) != len(goals):
        raise ValueError(f"Level {level_id!r} has {len(boxes)} boxes but {len(goals)} goals.")

    return SokobanLevel(
        level_id=level_id,
        grid=norm_lines,
        walls=frozenset(walls),
        goals=frozenset(goals),
        initial_state=SokobanState(player=player, boxes=tuple(boxes)),
    )


def parse_boxoban_file(path: Path | str, max_levels: Optional[int] = None) -> List[SokobanLevel]:
    """Parse a Boxoban text file whose puzzles are separated by ``; id`` lines."""

    path = Path(path)
    levels: list[SokobanLevel] = []
    current_id: Optional[str] = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        if current_id is not None and current_lines:
            levels.append(parse_sokoban_grid(current_lines, level_id=f"{path}:{current_id}"))
        current_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(";"):
            flush()
            current_id = line[1:].strip()
            if max_levels is not None and len(levels) >= max_levels:
                break
        elif current_id is not None and line:
            current_lines.append(line)
    if max_levels is None or len(levels) < max_levels:
        flush()
    return levels[:max_levels]


def iter_boxoban_files(path: Path | str) -> Iterator[Path]:
    path = Path(path)
    if path.is_file():
        yield path
    else:
        for fp in sorted(path.rglob("*.txt")):
            if not fp.name.lower().startswith("license"):
                yield fp


def load_boxoban_levels(path: Path | str, max_levels: Optional[int] = None) -> List[SokobanLevel]:
    """Load levels from a Boxoban file or directory."""

    levels: list[SokobanLevel] = []
    for fp in iter_boxoban_files(path):
        remaining = None if max_levels is None else max_levels - len(levels)
        if remaining is not None and remaining <= 0:
            break
        levels.extend(parse_boxoban_file(fp, max_levels=remaining))
    return levels


def render_state(level: SokobanLevel, state: SokobanState) -> str:
    grid = [["#" if (r, c) in level.walls else " " for c in range(level.width)] for r in range(level.height)]
    for r, c in level.goals:
        grid[r][c] = "."
    for r, c in state.boxes:
        grid[r][c] = "*" if (r, c) in level.goals else "$"
    pr, pc = state.player
    grid[pr][pc] = "+" if (pr, pc) in level.goals else "@"
    return "\n".join("".join(row) for row in grid)


def state_to_observation(level: SokobanLevel, state: SokobanState) -> np.ndarray:
    """Return four channels: wall, goal, box, player."""

    obs = np.zeros((level.height, level.width, 4), dtype=np.float32)
    for r, c in level.walls:
        obs[r, c, 0] = 1.0
    for r, c in level.goals:
        obs[r, c, 1] = 1.0
    for r, c in state.boxes:
        obs[r, c, 2] = 1.0
    pr, pc = state.player
    obs[pr, pc, 3] = 1.0
    return obs
