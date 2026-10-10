"""Conformation of an HP protein on the 2D lattice, without any GUI code.

This module holds everything phase 0 needs that is not drawing: the state of
the chain being built, the validation of the moves and the reading/writing of
the JSON file. It does not import Tkinter, so it can be tested with pytest
without opening a window.

Coordinates are pairs ``(x, y)`` of integers: ``x`` is the column
(``0 <= x < width``) and ``y`` is the row (``0 <= y < height``).
"""

import json
from enum import Enum
from pathlib import Path

Coord = tuple[int, int]

# The two residue types of the HP model.
RESIDUE_TYPES = ("H", "P")


class MoveError(Enum):
    """Reason why a residue cannot be placed on a cell."""

    CHAIN_COMPLETE = "chain_complete"
    OUTSIDE_GRID = "outside_grid"
    OCCUPIED = "occupied"
    NOT_ADJACENT = "not_adjacent"


def are_adjacent(a: Coord, b: Coord) -> bool:
    """True if the two cells are neighbours on the lattice (up/down/left/right)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


class Chain:
    """Chain under construction: positions of the residues and their types.

    The residues are stored in chain order. A residue whose type has not been
    assigned yet has type ``None``.
    """

    def __init__(self, width: int, height: int, length: int) -> None:
        self.width = width
        self.height = height
        self.length = length
        self.positions: list[Coord] = []
        self.types: list[str | None] = []

    # ------------------------------------------------------------------
    # Step 1: shape of the chain

    def contains(self, cell: Coord) -> bool:
        """True if the cell is inside the grid."""
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def check_move(self, cell: Coord) -> MoveError | None:
        """Returns why the next residue cannot go on `cell`, or None if it can."""
        if self.is_complete:
            return MoveError.CHAIN_COMPLETE
        if not self.contains(cell):
            return MoveError.OUTSIDE_GRID
        if cell in self.positions:
            return MoveError.OCCUPIED
        # The first residue can go anywhere.
        if self.positions and not are_adjacent(cell, self.positions[-1]):
            return MoveError.NOT_ADJACENT
        return None

    def add(self, cell: Coord) -> MoveError | None:
        """Places the next residue on `cell` if the move is valid.

        Returns None on success, otherwise the reason of the refusal (and the
        chain is left unchanged).
        """
        error = self.check_move(cell)
        if error is None:
            self.positions.append(cell)
            self.types.append(None)
        return error

    def undo(self) -> bool:
        """Removes the last residue. Returns False if the chain was empty."""
        if not self.positions:
            return False
        self.positions.pop()
        self.types.pop()
        return True

    def clear(self) -> None:
        """Removes all the residues."""
        self.positions.clear()
        self.types.clear()

    @property
    def is_complete(self) -> bool:
        """True when the chain has exactly `length` residues."""
        return len(self.positions) == self.length

    # ------------------------------------------------------------------
    # Step 2: H/P types

    def index_at(self, cell: Coord) -> int | None:
        """Index along the chain of the residue on `cell`, or None if it is free."""
        try:
            return self.positions.index(cell)
        except ValueError:
            return None

    def set_type(self, index: int, residue_type: str) -> None:
        """Assigns (or changes) the type of the residue at `index`."""
        if residue_type not in RESIDUE_TYPES:
            raise ValueError(f"invalid residue type: {residue_type!r}")
        self.types[index] = residue_type

    @property
    def is_fully_typed(self) -> bool:
        """True when the chain is complete and every residue has a type."""
        return self.is_complete and all(t is not None for t in self.types)

    @property
    def sequence(self) -> str:
        """HP sequence in chain order. Requires every residue to have a type."""
        if not self.is_fully_typed:
            raise ValueError("the chain is not complete or some residue has no type")
        return "".join(self.types)


def default_coordinates(width: int, height: int, length: int) -> list[Coord]:
    """Coordinates of a default conformation that fits in the grid.

    The chain fills the grid row by row, from left to right on even rows and
    from right to left on odd ones, so that consecutive residues are always
    adjacent. Used when the user does not draw the conformation by hand.
    """
    if length > width * height:
        raise ValueError(
            f"a chain of {length} residues does not fit in a {width}x{height} grid"
        )
    coordinates = []
    for i in range(length):
        y, offset = divmod(i, width)
        x = offset if y % 2 == 0 else width - 1 - offset
        coordinates.append((x, y))
    return coordinates


def validate_conformation(
    sequence: str, coordinates: list[Coord], width: int, height: int
) -> None:
    """Raises ValueError if the conformation is not a valid one for the grid.

    A valid conformation has one coordinate per residue, a sequence made only
    of H and P, and is a self-avoiding walk inside the grid.
    """
    if len(sequence) != len(coordinates):
        raise ValueError(
            f"{len(sequence)} residues in the sequence but {len(coordinates)} coordinates"
        )
    if not sequence:
        raise ValueError("the conformation is empty")
    if set(sequence) - set(RESIDUE_TYPES):
        raise ValueError(f"the sequence must contain only H and P: {sequence!r}")

    # Replaying the walk on a Chain applies the same rules used while drawing.
    chain = Chain(width, height, len(sequence))
    for index, cell in enumerate(coordinates):
        error = chain.add(cell)
        if error is not None:
            raise ValueError(f"residue {index} at {cell}: {error.value}")


def save_conformation(
    path: str | Path,
    sequence: str,
    coordinates: list[Coord],
    energy: int | None = None,
) -> None:
    """Writes the conformation to a JSON file, creating the folder if needed.

    Format (see notes/code_info/initial_conformation.md)::

        {"sequence": "HPPH", "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]}

    If `energy` is given (phase 1 output) it is added as a third key.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The file is written by hand instead of with json.dump(indent=...), which
    # would put every single number of the coordinates on its own line.
    pairs = ", ".join(f"[{x}, {y}]" for x, y in coordinates)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{\n")
        f.write(f'  "sequence": {json.dumps(sequence)},\n')
        if energy is None:
            f.write(f'  "coordinates": [{pairs}]\n')
        else:
            f.write(f'  "coordinates": [{pairs}],\n')
            f.write(f'  "energy": {energy}\n')
        f.write("}\n")


def load_conformation(path: str | Path) -> tuple[str, list[Coord]]:
    """Reads a file written by save_conformation: (sequence, coordinates)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sequence = data["sequence"]
    coordinates = [(int(x), int(y)) for x, y in data["coordinates"]]
    return sequence, coordinates
