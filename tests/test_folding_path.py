"""Path of one conformation to the native state, from the exact oracle."""

import pytest

from hpfold import _core
from hpfold import folding_path as fp

SEQUENCE = "HPPHPH"
STRAIGHT = [(x, 0) for x in range(len(SEQUENCE))]


def key(sequence, coordinates):
    conformation, _ = fp._place(sequence, coordinates)
    return _core.canonical_key(conformation)


def test_the_path_goes_down_one_move_at_a_time_to_a_native_state():
    path = fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16)
    oracle = fp.oracle(SEQUENCE)

    assert path.frames[0] == STRAIGHT
    assert len(path.steps) == oracle.distance(key(SEQUENCE, STRAIGHT)) > 0
    assert path.start_energy == 0
    keys = [key(SEQUENCE, frame) for frame in path.frames]
    for i, step in enumerate(path.steps):
        # Every frame is one action away from the previous one. The slots of
        # successors() are in the frame of the canonical conformation, which
        # can be the mirror image of the one of the step: only the keys are
        # compared.
        assert keys[i + 1] in [k for _, k in _core.successors(len(SEQUENCE), keys[i])]
        assert oracle.distance(keys[i + 1]) == step.distance == len(path.steps) - i - 1
        assert step.energy == fp._place(SEQUENCE, step.coordinates)[0].energy()
        assert (step.residue, _core.MOVE_TYPE_NAMES.index(step.move_type)) == \
            _core.action_table(len(SEQUENCE))[step.action]
    assert oracle.is_native(keys[-1])
    assert path.native == path.frames[-1]
    assert path.steps[-1].energy == path.target_energy == oracle.target_energy


def test_the_moves_keep_the_frame_of_the_input():
    # Far from the origin, with negative coordinates: the path is the same,
    # moved by the same shift, and no residue jumps more than two nodes.
    start = [(x - 40, y + 25) for x, y in STRAIGHT]
    path = fp.fold_to_native(SEQUENCE, start, max_length=16)
    reference = fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16)
    for frame, expected in zip(path.frames, reference.frames, strict=True):
        assert frame == [(x - 40, y + 25) for x, y in expected]
    for before, after in zip(path.frames, path.frames[1:]):
        assert all(abs(xa - xb) + abs(ya - yb) <= 2
                   for (xa, ya), (xb, yb) in zip(before, after))


def test_the_path_is_the_same_every_time():
    first = fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16)
    assert fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16) == first


def test_a_native_conformation_has_no_moves():
    native = fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16).native
    path = fp.fold_to_native(SEQUENCE, native, max_length=16)
    assert path.steps == []
    assert path.frames == [native] and path.native == native
    assert path.start_energy == path.target_energy


def test_chains_too_long_for_the_exact_oracle():
    with pytest.raises(ValueError):
        fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=5)


def test_format_path():
    path = fp.fold_to_native(SEQUENCE, STRAIGHT, max_length=16)
    text = fp.format_path(path)
    assert f"Moves to the nearest native state: {len(path.steps)}" in text
    assert path.steps[0].move_type in text
