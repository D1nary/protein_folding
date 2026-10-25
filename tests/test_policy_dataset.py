"""Phase 2: action space, canonical form, oracle, cleaning of the REMC
trajectories and dataset files."""

import copy
import csv
import random

import numpy as np
import pytest

from hpfold import _core, policy_dataset
from hpfold.config import load_config

# ----------------------------------------------------------------------
# Helpers


def random_state(n, steps, seed, sequence=None):
    """Canonical conformation reached from the straight chain with random actions."""
    rng = random.Random(seed)
    key = 0  # straight chain
    for _ in range(steps):
        key = rng.choice(_core.successors(n, key))[1]
    return _core.conformation_from_key(sequence or "P" * n, key)


def transformed(conformation, quarter_turns, flip, margin=3):
    """The conformation rotated, possibly mirrored, and translated on a lattice
    large enough for every move."""
    coordinates = np.array(conformation.coordinates)
    if flip:
        coordinates[:, 0] *= -1
    coordinates = policy_dataset.rotate_coordinates(coordinates, quarter_turns)
    coordinates = coordinates - coordinates.min(axis=0) + margin
    side = int(coordinates.max()) + margin + 1
    return _core.Conformation(
        conformation.sequence, [tuple(map(int, c)) for c in coordinates], side, side
    )


def next_keys(conformation):
    """Canonical key reached by every slot (None if the slot is not feasible)."""
    keys = []
    for move in _core.action_moves(conformation):
        if not move:
            keys.append(None)
            continue
        c = _core.Conformation(
            conformation.sequence, conformation.coordinates, conformation.width,
            conformation.height,
        )
        c.apply(move)
        assert c.is_valid()
        keys.append(_core.canonical_key(c))
    return keys


def as_set(moves):
    return {tuple(sorted(m)) for m in moves if m}


# ----------------------------------------------------------------------
# Action space


@pytest.mark.parametrize("n", [3, 4, 5, 8, 16])
def test_action_table_layout(n):
    table = _core.action_table(n)
    assert len(table) == _core.action_count(n) == 6 * n + 11
    assert len(set(table)) == len(table)
    for a, (residue, move_type) in enumerate(table):
        assert _core.action_index(n, residue, move_type) == a
    # End residues have the end moves, inner residues the others.
    names = _core.MOVE_TYPE_NAMES
    for residue, move_type in table:
        is_end_type = names[move_type].startswith("END") or names[move_type].startswith("PULL_END")
        assert is_end_type == (residue in (0, n - 1))


@pytest.mark.parametrize("n", [3, 8, 16])
def test_mirror_permutation_swaps_left_and_right(n):
    permutation = _core.mirror_permutation(n)
    table = _core.action_table(n)
    names = _core.MOVE_TYPE_NAMES
    swap = str.maketrans("LR", "RL")
    for a, b in enumerate(permutation):
        assert permutation[b] == a  # involution
        assert table[a][0] == table[b][0]  # same residue
        name, mirrored = names[table[a][1]], names[table[b][1]]
        prefix, _, letters = name.rpartition("_")
        if prefix in ("END", "PULL_NEXT", "PULL_PREV", "PULL_END"):
            assert mirrored == f"{prefix}_{letters.translate(swap)}"
        else:
            assert mirrored == name


@pytest.mark.parametrize("seed", range(40))
def test_action_moves_are_the_move_set(seed):
    # Every move of vshd_moves and pull_moves has exactly one slot, and every
    # slot holds one of those moves.
    c = random_state(10, steps=seed, seed=seed)
    reference = set()
    for k in range(len(c)):
        reference |= as_set(_core.vshd_moves(c, k)) | as_set(_core.pull_moves(c, k))
    moves = _core.action_moves(c)
    assert as_set(moves) == reference
    assert sum(1 for m in moves if m) == len(reference)


@pytest.mark.parametrize("seed", range(20))
def test_mask_marks_exactly_the_moves_that_keep_a_valid_chain(seed):
    sequence = "".join(random.Random(seed).choice("HP") for _ in range(12))
    c = random_state(12, steps=2 * seed, seed=seed, sequence=sequence)
    key = _core.canonical_key(c)
    oracle = _core.Oracle(_core.StateGraph(12, [key]), sequence)
    labels = oracle.label(np.array([key], dtype=np.uint64))
    moves = _core.action_moves(c)
    assert list(labels["mask"][0]) == [bool(m) for m in moves]
    # next_keys checks that the chain is valid (connected, self-avoiding)
    # after every feasible move.
    for move, target in zip(moves, next_keys(c)):
        assert (target is None) == (not move)
    assert all((d == _core.INVALID_ACTION) == (not m) for d, m in
               zip(labels["next_distance"][0], labels["mask"][0]))


# ----------------------------------------------------------------------
# Canonical form and symmetries


@pytest.mark.parametrize("seed", range(20))
def test_canonical_key_is_invariant_under_the_symmetries_of_the_lattice(seed):
    c = random_state(11, steps=3 * seed, seed=seed)
    key = _core.canonical_key(c)
    for quarter_turns in range(4):
        for flip in (False, True):
            for margin in (2, 5):
                assert _core.canonical_key(transformed(c, quarter_turns, flip, margin)) == key
    assert _core.turns(_core.conformation_from_key("P" * 11, key)) == _core.canonical_turns(c)
    assert _core.turns_key(_core.key_turns(key, 11)) == key


@pytest.mark.parametrize("n, count", [(3, 2), (5, 13), (8, 272), (10, 2034)])
def test_enumeration_counts_every_walk_once(n, count):
    # Self-avoiding walks of n-1 steps (4, 100, 2172, 16268), divided by 4
    # rotations and 2 reflections; only the straight chain is its own mirror.
    keys = _core.enumerate_states(n)
    assert len(keys) == count == len(np.unique(keys))
    assert all(_core.canonical_key(_core.conformation_from_key("P" * n, int(k))) == k
               for k in keys)


@pytest.mark.parametrize("seed", range(20))
def test_rotations_keep_the_actions_and_reflections_permute_them(seed):
    n = 10
    c = random_state(n, steps=2 * seed + 1, seed=seed)
    permutation = _core.mirror_permutation(n)
    reference = next_keys(transformed(c, 0, False))
    for quarter_turns in range(4):
        assert next_keys(transformed(c, quarter_turns, False)) == reference
        mirrored = next_keys(transformed(c, quarter_turns, True))
        assert [mirrored[permutation[a]] for a in range(len(reference))] == reference


def test_mirror_of_the_records_matches_the_mirrored_chain():
    n = 9
    sequence = "HPHHPPHPH"
    keys = np.unique([_core.canonical_key(random_state(n, 15, s)) for s in range(10)])
    oracle = _core.Oracle(_core.StateGraph(n, _core.enumerate_states(n)), sequence)
    batch = {"turns": policy_dataset.keys_to_turns(keys, n), **oracle.label(keys)}
    mirrored = policy_dataset.mirror(batch, _core.mirror_permutation(n))
    for i in range(len(keys)):
        chain = _core.conformation_from_turns(sequence, [int(t) for t in mirrored["turns"][i]])
        assert list(mirrored["mask"][i]) == [bool(m) for m in _core.action_moves(chain)]
    again = policy_dataset.mirror(mirrored, _core.mirror_permutation(n))
    for name in policy_dataset.MIRRORED_FIELDS:
        assert np.array_equal(again[name], batch[name])


def test_turns_to_coordinates_matches_the_cpp_decoding():
    n = 12
    keys = np.unique([_core.canonical_key(random_state(n, 20, s)) for s in range(15)])
    turns = policy_dataset.keys_to_turns(keys, n)
    coordinates = policy_dataset.turns_to_coordinates(turns)
    for key, c in zip(keys, coordinates):
        reference = np.array(_core.conformation_from_key("P" * n, int(key)).coordinates)
        assert np.array_equal(c, reference - reference[0])


def test_contact_map_gives_the_energy():
    n = 12
    sequence = "HHPHPHHPPHHH"
    keys = np.unique([_core.canonical_key(random_state(n, 25, s)) for s in range(15)])
    contacts = policy_dataset.contact_map(policy_dataset.turns_to_coordinates(policy_dataset.keys_to_turns(keys, n)))
    h = np.array([c == "H" for c in sequence])
    for key, m in zip(keys, contacts):
        energy = _core.conformation_from_key(sequence, int(key)).energy()
        assert -int((m & h[:, None] & h[None, :]).sum()) // 2 == energy


# ----------------------------------------------------------------------
# Oracle


@pytest.fixture(scope="module")
def exact_oracle():
    n = 9
    graph = _core.StateGraph(n, _core.enumerate_states(n))
    return _core.Oracle(graph, "HPHPPHHPH")


def test_distances_satisfy_the_shortest_path_equations(exact_oracle):
    n = 9
    graph_keys = _core.enumerate_states(n)
    distances = exact_oracle.distances
    # Every state can reach the native state (pull moves are complete).
    assert (distances >= 0).all()
    position = {int(k): i for i, k in enumerate(graph_keys)}
    for i, key in enumerate(graph_keys):
        key = int(key)
        native = exact_oracle.energy(key) == exact_oracle.target_energy
        assert (distances[i] == 0) == native
        if not native:
            best = min(distances[position[s]] for _, s in _core.successors(n, key) if s != key)
            assert best == distances[i] - 1


def test_the_move_graph_is_directed():
    # Some end pulls (PULL_END_LL, PULL_END_RR) cannot be undone in one move:
    # the distances are computed along the reversed edges, never assuming
    # that the graph is symmetric.
    n = 6
    one_way = 0
    for key in _core.enumerate_states(n):
        for _, target in _core.successors(n, int(key)):
            if target != key and int(key) not in [s for _, s in _core.successors(n, target)]:
                one_way += 1
    assert one_way > 0


@pytest.mark.parametrize("seed", range(10))
def test_oracle_paths_end_in_a_native_state(exact_oracle, seed):
    n = 9
    start = _core.canonical_key(random_state(n, 30, seed))
    path = exact_oracle.path(start, seed)
    keys, actions = list(path["keys"]), list(path["actions"])
    assert len(actions) == len(keys) - 1 == exact_oracle.distance(start)
    assert exact_oracle.is_native(int(keys[-1]))
    for key, action, nxt in zip(keys, actions, keys[1:]):
        assert (action, nxt) in _core.successors(n, int(key))
        assert exact_oracle.distance(int(nxt)) == exact_oracle.distance(int(key)) - 1


def test_graph_distances_are_upper_bounds(exact_oracle):
    n = 9
    rng = random.Random(0)
    visited = {0}
    key = 0
    for _ in range(3000):
        key = rng.choice(_core.successors(n, key))[1]
        visited.add(key)
    graph = _core.StateGraph(n, np.array(sorted(visited), dtype=np.uint64))
    oracle = _core.Oracle(graph, exact_oracle.sequence, exact_oracle.target_energy)
    for key, d in zip(graph.keys, oracle.distances):
        if d >= 0:
            assert d >= exact_oracle.distance(int(key))


def test_prune_trajectory_removes_loops_and_finds_shortcuts(exact_oracle):
    n = 9
    straight = 0
    a, b = (s for _, s in _core.successors(n, straight)[:2])
    path = exact_oracle.path(straight, 0)
    # straight -> a -> straight -> b -> straight -> (optimal path)
    trajectory = [straight, a, straight, b, straight] + [int(k) for k in path["keys"][1:]]
    pruned = _core.prune_trajectory(np.array(trajectory, dtype=np.uint64), exact_oracle)
    assert pruned["reached_native"]
    assert pruned["raw_length"] == len(trajectory) - 1
    assert pruned["loop_erased_length"] == len(path["actions"])
    assert pruned["shortest_length"] == exact_oracle.distance(straight)


# ----------------------------------------------------------------------
# REMC recording


def small_remc_params(seed=3):
    return _core.RemcParams(n_replicas=3, t_min=0.2, t_max=1.0, local_steps=50, n_cycles=40,
                            rho=0.5, seed=seed)


def test_recording_does_not_change_the_simulation():
    initial = policy_dataset.straight_chain("HPHPPHHPHH", 20, 20)
    plain = _core.run_remc(initial, small_remc_params())
    recorder = _core.SampleRecorder(100000, 5)
    recorded = _core.run_remc(initial, small_remc_params(), recorder)
    assert plain.best_energy == recorded.best_energy
    assert list(plain.final_energies) == list(recorded.final_energies)
    assert plain.best.coordinates == recorded.best.coordinates

    assert len(recorder.trajectories) == 3
    assert len(recorder.snapshots["key"]) == 3 * (40 // 5)
    for trajectory in recorder.trajectories:
        assert trajectory[0] == 0  # straight chain
        for key, nxt in zip(trajectory, trajectory[1:]):
            assert nxt in [s for _, s in _core.successors(10, int(key))]


def test_remc_trajectories_get_shorter_at_every_cleaning_step():
    n = 9
    sequence = "HPHPPHHPH"
    initial = policy_dataset.straight_chain(sequence, 20, 20)
    recorder = _core.SampleRecorder(100000, 5)
    _core.run_remc(initial, small_remc_params(seed=11), recorder)
    graph = _core.StateGraph(n, _core.enumerate_states(n))
    oracle = _core.Oracle(graph, sequence)
    reached = 0
    for trajectory in recorder.trajectories:
        pruned = _core.prune_trajectory(trajectory, oracle)
        if not pruned["reached_native"]:
            continue
        reached += 1
        assert (pruned["raw_length"] >= pruned["loop_erased_length"] >= pruned["shortest_length"]
                >= oracle.distance(int(trajectory[0])))
        keys = list(pruned["path"]["keys"])
        assert oracle.is_native(int(keys[-1]))
        for key, action, nxt in zip(keys, pruned["path"]["actions"], keys[1:]):
            assert (action, nxt) in _core.successors(n, int(key))
    assert reached > 0


# ----------------------------------------------------------------------
# Dataset files


def small_config(directory, exact):
    config = copy.deepcopy(load_config("config/default.toml"))
    config["protein"]["length"] = 9
    config["protein"]["sequence"] = "HPHPPHHPH"
    config["lattice"] = {"width": 20, "height": 20}
    config["output"]["policy_dataset_dir"] = str(directory)
    d = config["policy_dataset"]
    d.update(n_sequences=6, decoys_per_temperature=4, exact_max_length=9 if exact else 0,
             max_recorded_steps=20000)
    d["remc"].update(n_replicas=3, local_steps=50, n_cycles=40, save_every=4)
    return config


@pytest.fixture(scope="module", params=["exact", "graph"])
def built(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp(f"policy_dataset_{request.param}")
    config = small_config(directory, request.param == "exact")
    policy_dataset.build_dataset(config, verbose=False)
    return directory, config


def all_splits(directory):
    return {s: policy_dataset.load_split(directory, s) for s in policy_dataset.SPLITS
            if (directory / f"{s}.npz").exists()}


def test_files_and_metadata(built):
    directory, config = built
    meta = policy_dataset.load_meta(directory)
    assert meta["action_count"] == _core.action_count(9)
    assert meta["oracle"] == ("exact" if config["policy_dataset"]["exact_max_length"] >= 9 else "graph")
    sequences = policy_dataset.load_sequences(directory)
    assert len(sequences["sequence"]) == 6
    assert sequences["sequence"][0] == "HPHPPHHPH"
    assert (sequences["e_min"] < 0).all()
    for split, data in all_splits(directory).items():
        assert len(data["seq_id"]) == meta["records"][split]
        assert data["mask"].shape == data["next_distance"].shape == (len(data["seq_id"]), 65)


def test_splits_do_not_share_sequences(built):
    directory, _ = built
    sequences = policy_dataset.load_sequences(directory)
    seen = set()
    for name, data in all_splits(directory).items():
        ids = set(np.unique(data["seq_id"]).tolist())
        assert not ids & seen
        seen |= ids
        assert all(policy_dataset.SPLITS[sequences["split"][i]] == name for i in ids)
    assert seen == set(range(len(sequences["sequence"])))
    # A sequence and its reverse are the same chain: only one of them is used.
    groups = [min(q, q[::-1]) for q in sequences["sequence"]]
    assert len(set(groups)) == len(groups)


def test_masks_are_those_of_the_cpp_move_set(built):
    directory, _ = built
    for data in all_splits(directory).values():
        assert np.array_equal(data["mask"], data["next_distance"] != _core.INVALID_ACTION)
        for i in range(0, len(data["seq_id"]), 7):
            chain = _core.conformation_from_turns("P" * 9, [int(t) for t in data["turns"][i]])
            assert list(data["mask"][i]) == [bool(m) for m in _core.action_moves(chain)]


def test_every_state_has_an_optimal_action_or_is_native(built):
    directory, _ = built
    sequences = policy_dataset.load_sequences(directory)
    for data in all_splits(directory).values():
        optimal = policy_dataset.optimal_actions(data["next_distance"], data["distance"])
        assert (data["distance"] >= 0).all()
        assert (optimal.any(axis=1) == (data["distance"] > 0)).all()
        assert ((data["distance"] == 0) ==
                (data["energy"] == sequences["e_min"][data["seq_id"]])).all()
        soft = policy_dataset.soft_targets(data["next_distance"], data["distance"])
        assert np.allclose(soft.sum(axis=1), data["distance"] > 0)


def test_every_path_ends_in_the_native_state(built):
    directory, _ = built
    sequences = policy_dataset.load_sequences(directory)
    for data in all_splits(directory).values():
        count = 0
        for kind, records, actions in policy_dataset.paths(data):
            count += 1
            last = records[-1]
            assert data["distance"][last] == 0
            assert data["energy"][last] == sequences["e_min"][data["seq_id"][last]]
            assert actions[-1] == -1
            for r, action, nxt in zip(records, actions, records[1:]):
                key = _core.turns_key([int(t) for t in data["turns"][r]])
                target = _core.turns_key([int(t) for t in data["turns"][nxt]])
                assert (int(action), target) in _core.successors(9, key)
                assert data["next_distance"][r, action] == data["distance"][nxt]
                if policy_dataset.PATH_KINDS[kind] == "oracle":
                    assert data["distance"][nxt] == data["distance"][r] - 1
        assert count > 0


def test_trajectory_lengths_shrink_with_cleaning(built):
    directory, _ = built
    with open(directory / "trajectories.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows
    reached = [r for r in rows if r["reached_native"] == "1"]
    assert reached
    for r in reached:
        raw, loop, short, graph, exact = (int(r[k]) for k in (
            "raw_length", "loop_erased_length", "shortest_length", "graph_length",
            "exact_length"))
        assert raw >= loop >= short >= graph >= 0
        assert exact == -1 or graph >= exact


def test_augmentation_mirrors_about_half_of_the_records(built):
    directory, _ = built
    data = policy_dataset.load_split(directory, "train")
    permutation = policy_dataset.load_meta(directory)["mirror_permutation"]
    out = policy_dataset.augment(data, permutation, np.random.default_rng(0))
    changed = (out["turns"] != data["turns"]).any(axis=1)
    straight = (data["turns"] == 0).all(axis=1)
    assert 0.3 < changed[~straight].mean() < 0.7
    assert np.array_equal(out["distance"], data["distance"])
    grid = policy_dataset.lattice_grid(policy_dataset.turns_to_coordinates(out["turns"]), out["hp"])
    assert grid.shape == (len(data["turns"]), 3, 9, 9)
    assert np.array_equal(grid[:, 0].sum(axis=(1, 2)), out["hp"].sum(axis=1))
