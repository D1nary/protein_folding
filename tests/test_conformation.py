"""Tests of the phase 0 logic: chain state, move validation and JSON file.

None of these tests opens a window.
"""

import json

import pytest

from hpfold.conformation import (
    Chain,
    MoveError,
    are_adjacent,
    default_coordinates,
    load_conformation,
    save_conformation,
    validate_conformation,
)

# A 2x2 square: the shortest closed-looking walk, used by several tests.
SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]


def make_chain(cells, width=4, height=4, length=4):
    chain = Chain(width, height, length)
    for cell in cells:
        assert chain.add(cell) is None
    return chain


# ----------------------------------------------------------------------
# Step 1: shape of the chain


def test_are_adjacent():
    assert are_adjacent((1, 1), (1, 2))
    assert are_adjacent((1, 1), (0, 1))
    assert not are_adjacent((1, 1), (2, 2))  # diagonal
    assert not are_adjacent((1, 1), (1, 1))  # same cell
    assert not are_adjacent((1, 1), (1, 3))  # two cells away


def test_first_residue_can_go_anywhere_in_the_grid():
    for cell in [(0, 0), (3, 3), (2, 1)]:
        assert Chain(4, 4, 4).add(cell) is None


@pytest.mark.parametrize("cell", [(-1, 0), (0, -1), (4, 0), (0, 4)])
def test_cell_outside_grid_is_refused(cell):
    chain = Chain(4, 4, 4)
    assert chain.add(cell) is MoveError.OUTSIDE_GRID
    assert chain.positions == []


def test_occupied_cell_is_refused():
    chain = make_chain([(0, 0), (1, 0)])
    assert chain.add((0, 0)) is MoveError.OCCUPIED
    assert chain.add((1, 0)) is MoveError.OCCUPIED
    assert chain.positions == [(0, 0), (1, 0)]


@pytest.mark.parametrize("cell", [(1, 1), (2, 0), (3, 3)])
def test_non_adjacent_cell_is_refused(cell):
    chain = make_chain([(0, 0)])
    assert chain.add(cell) is MoveError.NOT_ADJACENT
    assert chain.positions == [(0, 0)]


def test_adjacency_is_checked_against_the_last_residue_only():
    chain = make_chain([(0, 0), (1, 0)])
    # Adjacent to the first residue but not to the last one.
    assert chain.add((0, 1)) is MoveError.NOT_ADJACENT


def test_chain_cannot_exceed_its_length():
    chain = make_chain(SQUARE[:3], length=3)
    assert chain.is_complete
    assert chain.add((0, 1)) is MoveError.CHAIN_COMPLETE
    assert len(chain.positions) == 3


def test_undo_removes_the_last_residue():
    chain = make_chain(SQUARE)
    assert chain.undo()
    assert chain.positions == SQUARE[:3]
    assert not chain.is_complete


def test_undo_on_empty_chain():
    assert not Chain(4, 4, 4).undo()


def test_clear():
    chain = make_chain(SQUARE)
    chain.set_type(0, "H")
    chain.clear()
    assert chain.positions == []
    assert chain.types == []


# ----------------------------------------------------------------------
# Step 2: H/P types


def test_index_at():
    chain = make_chain(SQUARE)
    assert chain.index_at((1, 1)) == 2
    assert chain.index_at((3, 3)) is None


def test_types_can_be_assigned_and_changed():
    chain = make_chain(SQUARE)
    assert chain.types == [None] * 4
    assert not chain.is_fully_typed

    for index, residue_type in enumerate("HPPH"):
        chain.set_type(index, residue_type)
    assert chain.is_fully_typed
    assert chain.sequence == "HPPH"

    chain.set_type(1, "H")
    assert chain.sequence == "HHPH"


def test_invalid_type_is_refused():
    chain = make_chain(SQUARE)
    with pytest.raises(ValueError):
        chain.set_type(0, "X")


def test_sequence_requires_all_types():
    chain = make_chain(SQUARE)
    chain.set_type(0, "H")
    with pytest.raises(ValueError):
        chain.sequence


def test_incomplete_chain_is_not_fully_typed():
    chain = make_chain(SQUARE[:2])
    chain.set_type(0, "H")
    chain.set_type(1, "P")
    assert not chain.is_fully_typed


def test_undo_keeps_the_types_of_the_other_residues():
    chain = make_chain(SQUARE)
    for index in range(4):
        chain.set_type(index, "H")
    chain.undo()
    assert chain.add((0, 1)) is None
    assert chain.types == ["H", "H", "H", None]


# ----------------------------------------------------------------------
# Default conformation and validation


@pytest.mark.parametrize(
    "width, height, length", [(8, 8, 16), (8, 8, 64), (3, 5, 7), (1, 4, 4), (4, 1, 4)]
)
def test_default_coordinates_are_a_valid_walk(width, height, length):
    coordinates = default_coordinates(width, height, length)
    assert len(coordinates) == length
    validate_conformation("H" * length, coordinates, width, height)


def test_default_coordinates_chain_too_long():
    with pytest.raises(ValueError):
        default_coordinates(2, 2, 5)


def test_validate_accepts_a_valid_conformation():
    validate_conformation("HPPH", SQUARE, 2, 2)


@pytest.mark.parametrize(
    "sequence, coordinates",
    [
        ("HPP", SQUARE),  # length mismatch
        ("", []),  # empty
        ("HPXH", SQUARE),  # invalid residue type
        ("HPPH", [(0, 0), (1, 0), (1, 1), (1, 0)]),  # not self-avoiding
        ("HPPH", [(0, 0), (1, 0), (1, 1), (0, 0)]),  # not self-avoiding
        ("HPPH", [(0, 0), (1, 1), (1, 0), (0, 1)]),  # not connected
        ("HPPH", [(0, 0), (1, 0), (2, 0), (2, 1)]),  # outside the 2x2 grid
    ],
)
def test_validate_refuses_invalid_conformations(sequence, coordinates):
    with pytest.raises(ValueError):
        validate_conformation(sequence, coordinates, 2, 2)


# ----------------------------------------------------------------------
# JSON file


def test_save_writes_the_documented_format(tmp_path):
    # The parent folder does not exist yet: it must be created.
    path = tmp_path / "input" / "conformation.json"
    save_conformation(path, "HPPH", SQUARE)

    assert json.loads(path.read_text()) == {
        "sequence": "HPPH",
        "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]],
    }


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "conformation.json"
    save_conformation(path, "HPPH", SQUARE)
    assert load_conformation(path) == ("HPPH", SQUARE)
