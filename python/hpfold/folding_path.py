"""Shortest folding path of one conformation to the native state.

Used after the judgement of the decoy discriminator (phase
``discriminator_score``): the model does not take part, the path comes from
the exact oracle of the policy dataset (cpp/include/hp/state_graph.hpp). The
oracle runs a breadth-first search on the graph of all the conformations, from
the native states (all the conformations at the lowest energy) backwards, so
it knows how many moves every conformation is from the nearest native state.

From the input conformation the path goes down one optimal move at a time,
from distance d to d - 1, until d = 0. The moves are applied to the chain as
it was given: residues that a move does not touch keep their coordinates, so
the frames of the path can be animated one after the other.
"""

import functools
from dataclasses import dataclass

from hpfold import _core
from hpfold.conformation import Coord

# Free nodes left around the chain when the moves are computed. No move takes
# a residue more than two nodes away from the box that contains the chain
# (free_lattice_side in cpp/include/hp/canonical.hpp): with this margin the
# lattice behaves as if it were infinite.
MARGIN = 2


@dataclass
class FoldingStep:
    """One move of the path and the conformation it leads to."""

    action: int                 # slot of the move (actions.hpp)
    move_type: str              # name of the type of the move
    residue: int                # residue of the slot
    coordinates: list[Coord]    # chain after the move
    energy: int                 # energy after the move
    distance: int               # moves left to the native state after this one


@dataclass
class FoldingPath:
    """Path from a conformation to the nearest native state."""

    sequence: str
    start: list[Coord]
    start_energy: int
    target_energy: int          # lowest energy of the sequence
    n_native: int               # canonical conformations at the lowest energy
    steps: list[FoldingStep]

    @property
    def native(self) -> list[Coord]:
        """Native state reached at the end of the path."""
        return self.steps[-1].coordinates if self.steps else self.start

    @property
    def frames(self) -> list[list[Coord]]:
        """Conformations of the path, the input first and the native state last."""
        return [self.start] + [step.coordinates for step in self.steps]


@functools.cache
def state_graph(n: int) -> _core.StateGraph:
    """Graph of all the conformations of a chain of n residues.

    It does not depend on the sequence: it is built once (about 5 s for
    n = 16) and kept for the whole session.
    """
    return _core.StateGraph(n, _core.enumerate_states(n))


@functools.lru_cache(maxsize=8)
def oracle(sequence: str) -> _core.Oracle:
    """Exact oracle of the sequence (about 1.5 s for 16 residues)."""
    return _core.Oracle(state_graph(len(sequence)), sequence)


@functools.cache
def _action_table(n: int) -> list[tuple[int, int]]:
    return _core.action_table(n)


def _place(sequence: str, coordinates: list[Coord]) -> tuple[_core.Conformation, Coord]:
    """The chain on a lattice with MARGIN free nodes around it.

    Returns the conformation and the shift (dx, dy) that takes its
    coordinates back to the ones given.
    """
    dx = min(x for x, _ in coordinates) - MARGIN
    dy = min(y for _, y in coordinates) - MARGIN
    shifted = [(x - dx, y - dy) for x, y in coordinates]
    width = max(x for x, _ in shifted) + 1 + MARGIN
    height = max(y for _, y in shifted) + 1 + MARGIN
    return _core.Conformation(sequence, shifted, width, height), (dx, dy)


def fold_to_native(sequence: str, coordinates: list[Coord], max_length: int) -> FoldingPath:
    """Shortest path of moves from the conformation to a native state.

    At every step the optimal moves are the ones that bring the distance from
    d to d - 1. When there are more of them, the one that lowers the energy
    the most is taken, then the one of the lowest slot: the path is always
    the same for the same input.

    Raises ValueError if the chain has more than `max_length` residues: the
    graph of all the conformations would be too large for the exact oracle.
    """
    n = len(sequence)
    if n > max_length:
        raise ValueError(
            f"the path to the native state is computed only for chains of up to "
            f"{max_length} residues (policy_dataset.exact_max_length), "
            f"this one has {n}"
        )
    the_oracle = oracle(sequence)
    conformation, _ = _place(sequence, coordinates)
    distance = the_oracle.distance(_core.canonical_key(conformation))
    if distance < 0:
        raise ValueError("the native state cannot be reached from this conformation")

    table = _action_table(n)
    current = list(coordinates)
    steps = []
    while distance > 0:
        conformation, (dx, dy) = _place(sequence, current)
        best = None
        for action, move in enumerate(_core.action_moves(conformation)):
            if not move:
                continue
            after = _core.Conformation(
                sequence, conformation.coordinates, conformation.width, conformation.height
            )
            after.apply(move)
            if the_oracle.distance(_core.canonical_key(after)) != distance - 1:
                continue
            if best is None or after.energy() < best[1].energy():
                best = (action, after)
        if best is None:
            raise RuntimeError(f"no optimal move from a state at distance {distance}")
        action, after = best
        current = [(x + dx, y + dy) for x, y in after.coordinates]
        distance -= 1
        residue, move_type = table[action]
        steps.append(FoldingStep(
            action=action,
            move_type=_core.MOVE_TYPE_NAMES[move_type],
            residue=residue,
            coordinates=current,
            energy=after.energy(),
            distance=distance,
        ))

    return FoldingPath(
        sequence=sequence,
        start=list(coordinates),
        start_energy=_place(sequence, coordinates)[0].energy(),
        target_energy=the_oracle.target_energy,
        n_native=len(the_oracle.native_keys),
        steps=steps,
    )


def format_path(path: FoldingPath) -> str:
    """Text of the path, printed on the terminal."""
    lines = [
        f"Lowest energy of the sequence (exact): {path.target_energy}"
        f"   native states: {path.n_native}",
        f"Moves to the nearest native state: {len(path.steps)}",
        f"  start: energy {path.start_energy}",
    ]
    for i, step in enumerate(path.steps, start=1):
        lines.append(
            f"  {i}. {step.move_type} at residue {step.residue}: "
            f"energy {step.energy}, {step.distance} moves left"
        )
    lines.append(f"Native state: {[list(c) for c in path.native]}")
    return "\n".join(lines)
