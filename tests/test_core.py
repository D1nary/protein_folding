"""C++ core (hpfold._core): energy of the HP model, VSHD moves and pull moves."""

import random

import pytest

from hpfold import _core
from hpfold.conformation import are_adjacent, default_coordinates, validate_conformation


def reference_energy(sequence, coordinates):
    """Energy computed directly from the definition, without the lattice."""
    energy = 0
    for i in range(len(sequence)):
        for j in range(i + 2, len(sequence)):
            if sequence[i] == sequence[j] == "H" and are_adjacent(coordinates[i], coordinates[j]):
                energy -= 1
    return energy


def straight(sequence, width, height, y=0):
    return _core.Conformation(sequence, [(x, y) for x in range(len(sequence))], width, height)


# ----------------------------------------------------------------------
# Conformation and energy


def test_invalid_conformations_are_rejected():
    with pytest.raises(ValueError):
        _core.Conformation("HPX", [(0, 0), (1, 0), (2, 0)], 5, 5)  # not only H and P
    with pytest.raises(ValueError):
        _core.Conformation("HPH", [(0, 0), (1, 0)], 5, 5)  # one coordinate missing
    with pytest.raises(ValueError):
        _core.Conformation("HPH", [(3, 0), (4, 0), (5, 0)], 5, 5)  # outside the lattice
    with pytest.raises(ValueError):
        _core.Conformation("HPH", [(0, 0), (1, 0), (0, 0)], 5, 5)  # overlap
    with pytest.raises(ValueError):
        _core.Conformation("HPH", [(0, 0), (1, 0), (2, 1)], 5, 5)  # broken chain


def test_energy_of_a_straight_chain_is_zero():
    assert straight("HHHHHH", 6, 1).energy() == 0


def test_energy_counts_only_non_consecutive_hh_contacts():
    # Square: residues 0 and 3 are adjacent on the lattice but not in the chain.
    square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert _core.Conformation("HPPH", square, 2, 2).energy() == -1
    assert _core.Conformation("HHHH", square, 2, 2).energy() == -1
    assert _core.Conformation("HPPP", square, 2, 2).energy() == 0
    assert _core.Conformation("PHHP", square, 2, 2).energy() == 0


def test_energy_of_the_paper_example():
    # Figure 1 of Thachuk et al. (2007): ground state with energy -2.
    coordinates = [
        (0, 1), (0, 0), (1, 0), (1, 1), (2, 1), (2, 0), (3, 0), (3, 1),
        (3, 2), (3, 3), (2, 3), (2, 2), (1, 2), (1, 3), (0, 3), (0, 2),
    ]  # fmt: skip
    assert _core.Conformation("PPPHHPPPPPPHHPPP", coordinates, 4, 4).energy() == -2


def test_energy_matches_the_definition_on_a_compact_chain():
    # Serpentine filling a 4x4 lattice: many contacts of every kind.
    sequence = "HHPHHPHHHPHPHHHH"
    coordinates = default_coordinates(4, 4, 16)
    conformation = _core.Conformation(sequence, coordinates, 4, 4)
    assert conformation.energy() == reference_energy(sequence, coordinates) < 0


# ----------------------------------------------------------------------
# VSHD moves


def test_end_move_targets_are_the_free_nodes_around_the_neighbour():
    c = straight("HPH", 5, 3, y=1)  # (0,1) (1,1) (2,1)
    assert sorted(_core.vshd_moves(c, 0)) == [[(0, (1, 0))], [(0, (1, 2))]]


def test_corner_move():
    c = _core.Conformation("HPH", [(0, 0), (1, 0), (1, 1)], 3, 3)
    assert _core.vshd_moves(c, 1) == [[(1, (0, 1))]]
    # No corner move on a straight segment.
    assert _core.vshd_moves(straight("HPH", 3, 3, y=1), 1) == []


def test_crankshaft_move():
    # U-shaped bend on rows 1-2: residues 1 and 2 flip from row 2 to row 0.
    coordinates = [(1, 1), (1, 2), (2, 2), (2, 1)]
    c = _core.Conformation("HPPH", coordinates, 4, 4)
    flip = [(1, (1, 0)), (2, (2, 0))]
    # The same move is found from either of the two residues it displaces.
    assert flip in _core.vshd_moves(c, 1)
    assert flip in _core.vshd_moves(c, 2)

    c.apply(flip)
    assert c.coordinates == [(1, 1), (1, 0), (2, 0), (2, 1)]
    assert c.is_valid()


def test_crankshaft_needs_both_target_nodes_free():
    # Same bend against the wall: the targets would be outside the lattice.
    c = _core.Conformation("HPPH", [(1, 1), (1, 0), (2, 0), (2, 1)], 4, 2)
    assert _core.vshd_moves(c, 1) == []


# ----------------------------------------------------------------------
# Pull moves


def test_pull_moves_of_an_inner_residue():
    c = straight("HPPH", 5, 3, y=1)  # (0,1) (1,1) (2,1) (3,1)
    assert sorted(_core.pull_moves(c, 1)) == [
        # Towards residue 2: residue 0 follows into the fourth corner.
        [(1, (0, 0)), (2, (1, 0)), (3, (1, 1))],
        [(1, (0, 2)), (2, (1, 2)), (3, (1, 1))],
        # Towards residue 0: residues 2 and 3 are pulled along.
        [(1, (2, 0)), (0, (1, 0))],
        [(1, (2, 2)), (0, (1, 2))],
    ]


def test_pull_stops_as_soon_as_the_chain_is_connected():
    # Residue 3 is already adjacent to the node where residue 2 lands, so
    # residues 3, 4, 5 do not move.
    coordinates = [(0, 1), (1, 1), (2, 1), (2, 2), (3, 2), (4, 2)]
    c = _core.Conformation("HPPHPP", coordinates, 5, 4)
    assert [(1, (0, 2)), (2, (1, 2))] in _core.pull_moves(c, 1)


def test_pull_with_occupied_corner_is_a_corner_move_listed_once():
    c = _core.Conformation("HPH", [(0, 0), (1, 0), (1, 1)], 3, 3)
    # Both directions of the pull find the corner move 1 -> (0, 1).
    assert _core.pull_moves(c, 1).count([(1, (0, 1))]) == 1


def test_end_pull_move_advances_the_end_by_two_nodes():
    c = straight("HPH", 5, 1)  # (0,0) (1,0) (2,0), only one row
    # The only two free nodes in a row are (3,0) and (4,0), beyond residue 2.
    assert _core.pull_moves(c, 2) == [[(2, (4, 0)), (1, (3, 0)), (0, (2, 0))]]
    assert _core.pull_moves(c, 0) == []


def test_no_move_leaves_a_lattice_that_is_full():
    c = straight("HPHP", 4, 1)
    for k in range(4):
        assert _core.vshd_moves(c, k) == []
        assert _core.pull_moves(c, k) == []


# ----------------------------------------------------------------------
# Validity of the moves: connected chain, self-avoiding, inside the lattice


@pytest.mark.parametrize("generator", [_core.vshd_moves, _core.pull_moves])
@pytest.mark.parametrize("width, height", [(30, 30), (6, 5)])
def test_every_move_gives_a_valid_conformation(generator, width, height):
    """Random walk through the move set, checking every feasible move.

    The small lattice keeps the chain against the walls. Validity is checked
    with the Python code of phase 0, independent of the C++ one.
    """
    sequence = "HPHPPHHPHPPHPHHH"
    start = default_coordinates(width, height, len(sequence))
    conformation = _core.Conformation(sequence, start, width, height)
    rng = random.Random(0)
    applied = 0

    for _ in range(2000):
        k = rng.randrange(len(sequence))
        moves = generator(conformation, k)
        before = conformation.coordinates

        for move in moves:
            assert any(index == k for index, _ in move)
            undo = conformation.apply(move)
            after = conformation.coordinates
            validate_conformation(sequence, after, width, height)  # raises if not valid
            assert conformation.is_valid()
            assert conformation.energy() == reference_energy(sequence, after)
            assert after != before
            conformation.apply(undo)
            assert conformation.coordinates == before

        assert len({tuple(move) for move in moves}) == len(moves)  # no duplicates
        if moves:
            conformation.apply(rng.choice(moves))
            applied += 1

    assert applied > 500  # the walk really moved
