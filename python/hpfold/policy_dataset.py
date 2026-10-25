"""Phase 2 — generation of the labelled dataset for the folding policy.

Every record is a state (sequence, conformation) with the labels the policy
learns from: which actions are feasible, how many moves every action leaves to
the native state, and the energy change of every action. The labels come from
an oracle (breadth-first search on the graph of the moves), never from the
moves made by the REMC: the REMC only provides the states. See
notes/dataset.md for the format and the reasons behind it.

The heavy parts (action space, canonical keys, graph, oracle, cleaning of the
trajectories) are in the C++ core; this module chooses the states, splits the
sequences, writes the files and reads them back (``load_split``).
"""

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from hpfold import _core

FORMAT_VERSION = 1
SPLITS = ("train", "val", "test")
# Where a record comes from (field `source`).
SOURCES = ("remc_snapshot", "remc_pruned", "oracle_path", "perturbed")
# Kind of a path (field `path_kind`).
PATH_KINDS = ("oracle", "remc_pruned")
# Turn codes of the field `turns`, as in cpp/include/hp/canonical.hpp.
TURNS = ("FORWARD", "LEFT", "RIGHT")

TRAJECTORY_FIELDS = (
    "seq_id",
    "walker",
    "reached_native",
    "raw_length",
    "loop_erased_length",
    "shortest_length",
    "graph_length",
    "exact_length",
)


# ----------------------------------------------------------------------
# Sequences and splits


def candidate_sequences(n: int, h_fraction: float, rng: np.random.Generator, first=None):
    """Endless stream of distinct HP sequences of length n.

    A sequence and its reverse describe the same chain read from the other
    end, so only one of the two is produced: otherwise the same folding
    problem could end up both in the training and in the test split.
    ``first``, if given, comes out first.
    """
    seen = set()

    def new(sequence):
        group = min(sequence, sequence[::-1])
        if group in seen:
            return False
        seen.add(group)
        return True

    if first is not None and new(first):
        yield first
    while True:
        sequence = "".join(np.where(rng.random(n) < h_fraction, "H", "P"))
        if new(sequence):
            yield sequence


def assign_splits(count: int, fractions, rng: np.random.Generator) -> np.ndarray:
    """Split index (0 train, 1 val, 2 test) of each of `count` sequences.

    The split is by sequence: all the records of a sequence go to the same
    split. With at least three sequences, validation and test get at least
    one each.
    """
    total = sum(fractions)
    n_val = round(count * fractions[1] / total)
    n_test = round(count * fractions[2] / total)
    if count >= 3:
        n_val, n_test = max(n_val, 1), max(n_test, 1)
    n_train = max(count - n_val - n_test, 0)
    split = np.empty(count, dtype=np.uint8)
    order = rng.permutation(count)
    split[order[:n_train]] = 0
    split[order[n_train : n_train + n_val]] = 1
    split[order[n_train + n_val :]] = 2
    return split


# ----------------------------------------------------------------------
# Records of one sequence


def keys_to_turns(keys, n: int) -> np.ndarray:
    """Turns (S, n-2) of an array of canonical keys (inverse of turns_key)."""
    keys = np.asarray(keys, dtype=np.uint64).copy()
    turns = np.zeros((len(keys), n - 2), dtype=np.uint8)
    for i in range(n - 3, -1, -1):
        turns[:, i] = keys % np.uint64(3)
        keys //= np.uint64(3)
    return turns


def straight_chain(sequence: str, width: int, height: int) -> _core.Conformation:
    """Unfolded starting state of the REMC: a straight chain in the middle row."""
    n = len(sequence)
    if n > width:
        raise ValueError(f"a straight chain of {n} residues does not fit in width {width}")
    x0, y = (width - n) // 2, height // 2
    return _core.Conformation(sequence, [(x0 + i, y) for i in range(n)], width, height)


def sample_remc(sequence: str, seq_id: int, config: dict):
    """Runs the REMC of `[policy_dataset.remc]` and returns the recorder and the result."""
    lattice = config["lattice"]
    dataset = config["policy_dataset"]
    remc = dataset["remc"]
    params = _core.RemcParams(
        n_replicas=remc["n_replicas"],
        t_min=remc["t_min"],
        t_max=remc["t_max"],
        local_steps=remc["local_steps"],
        n_cycles=remc["n_cycles"],
        rho=remc["rho"],
        seed=remc["seed"] + seq_id,
    )
    recorder = _core.SampleRecorder(dataset["max_recorded_steps"], remc["save_every"])
    initial = straight_chain(sequence, lattice["width"], lattice["height"])
    result = _core.run_remc(initial, params, recorder)
    return recorder, result


def random_walk(n: int, key: int, steps: int, rng: np.random.Generator) -> int:
    """State reached from `key` with `steps` feasible actions chosen at random."""
    for _ in range(steps):
        options = _core.successors(n, key)
        key = options[rng.integers(len(options))][1]
    return key


def sequence_records(sequence: str, seq_id: int, config: dict, full_graph=None):
    """Records, paths and trajectory statistics of one sequence.

    With `full_graph` (the graph of all the states, for short chains) the
    oracle is exact; otherwise it works on the graph of the states visited by
    the REMC. Returns None if the lowest energy is 0: every state would be
    native and there would be nothing to learn.
    """
    n = len(sequence)
    dataset = config["policy_dataset"]
    rng = np.random.default_rng([dataset["seed"], seq_id])

    # 1. States visited by the REMC: walkers' trajectories and snapshots.
    recorder, result = sample_remc(sequence, seq_id, config)
    trajectories = recorder.trajectories
    snapshots = recorder.snapshots
    temperatures = list(result.temperatures)
    remc_graph = _core.StateGraph(n, np.unique(np.concatenate(trajectories + [snapshots["key"]])))

    # 2. Oracle: native states = all the states with the lowest energy.
    if full_graph is not None:
        oracle = _core.Oracle(full_graph, sequence)
        try:
            remc_oracle = _core.Oracle(remc_graph, sequence, oracle.target_energy)
        except ValueError:  # the REMC never reached the lowest energy
            remc_oracle = None
    else:
        oracle = remc_oracle = _core.Oracle(remc_graph, sequence)
    if oracle.target_energy == 0:
        return None

    # 3. Cleaning of the trajectories, measured at every step.
    trajectory_rows, pruned_paths = [], []
    for walker, trajectory in enumerate(trajectories):
        pruned = _core.prune_trajectory(trajectory, oracle)
        start = int(trajectory[0])
        graph_length = remc_oracle.distance(start) if remc_oracle is not None else -1
        exact_length = oracle.distance(start) if full_graph is not None else -1
        trajectory_rows.append(
            {
                "seq_id": seq_id,
                "walker": walker,
                "reached_native": int(pruned["reached_native"]),
                "raw_length": pruned["raw_length"],
                "loop_erased_length": pruned["loop_erased_length"],
                "shortest_length": pruned["shortest_length"],
                "graph_length": max(graph_length, -1),
                "exact_length": max(exact_length, -1),
            }
        )
        if pruned["reached_native"]:
            pruned_paths.append(pruned["path"])

    # 4. Decoys: snapshots of every replica, i.e. of every temperature.
    keys, sources, record_temperature = [], [], []
    position = {}

    def add(key, source, temperature=np.nan):
        key = int(key)
        if key not in position:
            position[key] = len(keys)
            keys.append(key)
            sources.append(source)
            record_temperature.append(temperature)

    decoys = []
    for replica, temperature in enumerate(temperatures):
        candidates = np.unique(snapshots["key"][snapshots["replica"] == replica])
        candidates = [int(k) for k in candidates if oracle.distance(int(k)) >= 0]
        take = min(dataset["decoys_per_temperature"], len(candidates))
        # Indices, not keys: a key may not fit in a signed 64-bit integer.
        for i in rng.choice(len(candidates), size=take, replace=False):
            add(candidates[i], SOURCES.index("remc_snapshot"), temperature)
            decoys.append(candidates[i])

    # 5. Paths to the native state: optimal ones from every decoy, and the
    #    cleaned REMC trajectories.
    paths = []
    for path in pruned_paths:
        for k in path["keys"]:
            add(k, SOURCES.index("remc_pruned"))
        paths.append((PATH_KINDS.index("remc_pruned"), path))
    for decoy in decoys:
        path = oracle.path(decoy, int(rng.integers(2**63)))
        for k in path["keys"]:
            add(k, SOURCES.index("oracle_path"))
        paths.append((PATH_KINDS.index("oracle"), path))

    # 6. Recovery states: path states pushed off course by a few random moves.
    path_keys = list(keys)
    count = round(dataset["perturb_fraction"] * len(path_keys))
    for i in rng.choice(len(path_keys), size=min(count, len(path_keys)), replace=False):
        steps = int(rng.integers(1, dataset["perturb_max_moves"] + 1))
        add(random_walk(n, path_keys[i], steps, rng), SOURCES.index("perturbed"))

    # 7. Labels; perturbed states whose distance is not known are dropped.
    labels = oracle.label(np.array(keys, dtype=np.uint64))
    keep = labels["distance"] >= 0
    if not keep[: len(path_keys)].all():
        raise RuntimeError("a state on a path has no known distance")
    new_index = np.cumsum(keep) - 1

    records = {
        "seq_id": np.full(int(keep.sum()), seq_id, dtype=np.int32),
        "turns": keys_to_turns(np.array(keys, dtype=np.uint64)[keep], n),
        "energy": labels["energy"][keep],
        "distance": labels["distance"][keep],
        "mask": labels["mask"][keep],
        "next_distance": labels["next_distance"][keep],
        "next_energy_delta": labels["next_energy_delta"][keep],
        "source": np.array(sources, dtype=np.uint8)[keep],
        "temperature": np.array(record_temperature, dtype=np.float32)[keep],
    }
    record_paths = [
        (kind, new_index[[position[int(k)] for k in path["keys"]]], np.asarray(path["actions"]))
        for kind, path in paths
    ]
    info = {
        "sequence": sequence,
        "e_min": oracle.target_energy,
        "n_native": len(oracle.native_keys),
        "remc_best_energy": result.best_energy,
        "exact": full_graph is not None,
        "visited_states": len(remc_graph),
    }
    return records, record_paths, trajectory_rows, info


# ----------------------------------------------------------------------
# Writing


def _concatenate(parts: list[dict]) -> dict:
    return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}


def _split_arrays(parts, n_actions: int) -> dict:
    """Arrays of one split: records (mask packed in bits) and paths."""
    records = _concatenate([p[0] for p in parts])
    records["mask"] = np.packbits(records["mask"], axis=1, bitorder="little")

    offsets, path_records, path_actions, kinds, path_seq = [0], [], [], [], []
    shift = 0
    for part_records, part_paths, _, _ in parts:
        for kind, indices, actions in part_paths:
            path_records.append(indices + shift)
            # actions[i] leads from record i to record i+1; -1 on the last one.
            path_actions.append(np.append(actions, -1))
            offsets.append(offsets[-1] + len(indices))
            kinds.append(kind)
            path_seq.append(part_records["seq_id"][0])
        shift += len(part_records["seq_id"])
    records["path_offsets"] = np.array(offsets, dtype=np.int64)
    records["path_records"] = np.concatenate(path_records).astype(np.int32)
    records["path_actions"] = np.concatenate(path_actions).astype(np.int16)
    records["path_kind"] = np.array(kinds, dtype=np.uint8)
    records["path_seq_id"] = np.array(path_seq, dtype=np.int32)
    return records


def _git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_dataset(config: dict, verbose: bool = True) -> Path:
    """Generates the dataset in `output.policy_dataset_dir` and returns the directory."""
    n = config["protein"]["length"]
    dataset = config["policy_dataset"]
    if n < 3 or n > _core.MAX_KEY_LENGTH:
        raise ValueError(f"protein.length must be between 3 and {_core.MAX_KEY_LENGTH}")

    full_graph = None
    if n <= dataset["exact_max_length"]:
        if verbose:
            print(f"Building the graph of all the states of {n} residues...")
        full_graph = _core.StateGraph(n, _core.enumerate_states(n))
        if verbose:
            print(f"  {len(full_graph)} states, {full_graph.edge_count} edges")

    rng = np.random.default_rng(dataset["seed"])
    first = config["protein"]["sequence"] if dataset["include_config_sequence"] else None
    if first is not None and len(first) != n:
        raise ValueError("protein.sequence must have protein.length residues")
    stream = candidate_sequences(n, dataset["h_fraction"], rng, first)

    parts, attempts = [], 0
    while len(parts) < dataset["n_sequences"]:
        attempts += 1
        if attempts > 20 * dataset["n_sequences"]:
            raise RuntimeError("too few sequences with a negative lowest energy")
        sequence = next(stream)
        part = sequence_records(sequence, len(parts), config, full_graph)
        if part is None:
            continue
        parts.append(part)
        if verbose:
            info = part[3]
            print(
                f"  [{len(parts)}/{dataset['n_sequences']}] {sequence}  E_min={info['e_min']}"
                f"  native={info['n_native']}  records={len(part[0]['seq_id'])}"
            )

    split = assign_splits(len(parts), dataset["split"], rng)
    out = Path(config["output"]["policy_dataset_dir"])
    out.mkdir(parents=True, exist_ok=True)
    n_actions = _core.action_count(n)

    counts = {}
    for s, name in enumerate(SPLITS):
        chosen = [p for p, k in zip(parts, split) if k == s]
        counts[name] = sum(len(p[0]["seq_id"]) for p in chosen)
        if chosen:
            np.savez_compressed(out / f"{name}.npz", **_split_arrays(chosen, n_actions))
        else:
            (out / f"{name}.npz").unlink(missing_ok=True)

    infos = [p[3] for p in parts]
    np.savez_compressed(
        out / "sequences.npz",
        seq_id=np.arange(len(parts), dtype=np.int32),
        sequence=np.array([i["sequence"] for i in infos]),
        hp=np.array([[c == "H" for c in i["sequence"]] for i in infos], dtype=np.uint8),
        split=split,
        e_min=np.array([i["e_min"] for i in infos], dtype=np.int16),
        n_native=np.array([i["n_native"] for i in infos], dtype=np.int32),
        remc_best_energy=np.array([i["remc_best_energy"] for i in infos], dtype=np.int16),
        exact=np.array([i["exact"] for i in infos], dtype=bool),
        visited_states=np.array([i["visited_states"] for i in infos], dtype=np.int32),
    )

    with open(out / "trajectories.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        for p in parts:
            writer.writerows(p[2])

    meta = {
        "format_version": FORMAT_VERSION,
        "length": n,
        "action_count": n_actions,
        "move_types": list(_core.MOVE_TYPE_NAMES),
        "action_table": [list(a) for a in _core.action_table(n)],
        "mirror_permutation": list(_core.mirror_permutation(n)),
        "turns": list(TURNS),
        "sources": list(SOURCES),
        "path_kinds": list(PATH_KINDS),
        "splits": list(SPLITS),
        "invalid_action": _core.INVALID_ACTION,
        "unknown_distance": _core.UNKNOWN_DISTANCE,
        "oracle": "exact" if full_graph is not None else "graph",
        "records": counts,
        "config": {"policy_dataset": dataset, "lattice": config["lattice"]},
        "git_commit": _git_commit(),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


def run(config: dict) -> None:
    """Builds the dataset and prints a summary."""
    out = build_dataset(config)
    meta = load_meta(out)
    print(f"Oracle:   {meta['oracle']}")
    print(f"Records:  {meta['records']}")
    with open(out / "trajectories.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["reached_native"] == "1"]
    if rows:
        for field in ("raw_length", "loop_erased_length", "shortest_length", "graph_length",
                      "exact_length"):
            values = [int(r[field]) for r in rows if int(r[field]) >= 0]
            if values:
                print(f"Trajectory {field:<19} mean {np.mean(values):10.1f}")
    print(f"Saved to: {out}")


# ----------------------------------------------------------------------
# Reading


def load_meta(dataset_dir) -> dict:
    with open(Path(dataset_dir) / "meta.json") as f:
        return json.load(f)


def load_sequences(dataset_dir) -> dict:
    with np.load(Path(dataset_dir) / "sequences.npz") as f:
        return {name: f[name] for name in f.files}


def load_split(dataset_dir, split: str) -> dict:
    """Arrays of one split, with the mask unpacked to bool (S, A).

    Also adds `hp`, the sequence of every record as 0/1 (S, N).
    """
    meta = load_meta(dataset_dir)
    with np.load(Path(dataset_dir) / f"{split}.npz") as f:
        data = {name: f[name] for name in f.files}
    data["mask"] = np.unpackbits(
        data["mask"], axis=1, count=meta["action_count"], bitorder="little"
    ).astype(bool)
    data["hp"] = load_sequences(dataset_dir)["hp"][data["seq_id"]]
    return data


def paths(data: dict):
    """The paths of a split, as (kind, record indices, actions) tuples."""
    offsets = data["path_offsets"]
    for p in range(len(offsets) - 1):
        a, b = offsets[p], offsets[p + 1]
        yield int(data["path_kind"][p]), data["path_records"][a:b], data["path_actions"][a:b]


def optimal_actions(next_distance: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Multi-label target (S, A): the actions that leave one move less.

    States at distance 0 (native) have no optimal action.
    """
    d = np.asarray(distance)[:, None]
    return (next_distance >= 0) & (next_distance == d - 1)


def soft_targets(next_distance, distance, next_energy_delta=None, temperature=None):
    """Distribution over the actions (S, A): uniform on the optimal ones, or,
    with a temperature, weighted by exp(-ΔE / temperature) among them."""
    optimal = optimal_actions(next_distance, distance)
    weight = optimal.astype(np.float64)
    if temperature is not None:
        weight *= np.exp(-np.asarray(next_energy_delta, dtype=np.float64) / temperature)
    total = weight.sum(axis=1, keepdims=True)
    return np.divide(weight, total, out=np.zeros_like(weight), where=total > 0)


# ----------------------------------------------------------------------
# Encodings of the conformation


def turns_to_coordinates(turns) -> np.ndarray:
    """Coordinates (S, N, 2) of the chains: residue 0 at the origin, first bond
    to the right, as conformation_from_turns in C++ (up to a translation)."""
    turns = np.atleast_2d(np.asarray(turns, dtype=np.int64))
    # Directions 0 +x, 1 +y, 2 -x, 3 -y: LEFT adds one quarter turn, RIGHT
    # removes one.
    step = np.where(turns == 1, 1, np.where(turns == 2, 3, 0))
    direction = np.concatenate(
        [np.zeros((len(turns), 1), dtype=np.int64), np.cumsum(step, axis=1) % 4], axis=1
    )
    unit = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    bonds = unit[direction]
    start = np.zeros((len(turns), 1, 2), dtype=np.int64)
    return np.concatenate([start, np.cumsum(bonds, axis=1)], axis=1)


def contact_map(coordinates) -> np.ndarray:
    """Contact matrix (S, N, N): residues adjacent on the lattice but not in
    the chain."""
    c = np.asarray(coordinates)
    distance = np.abs(c[:, :, None, :] - c[:, None, :, :]).sum(axis=-1)
    n = c.shape[1]
    far = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 1
    return (distance == 1) & far


def lattice_grid(coordinates, hp, size: int | None = None) -> np.ndarray:
    """Image of the lattice (S, 3, size, size), the chain centred.

    Channels: H residues, P residues, position along the chain (i+1)/N.
    `size` defaults to N, enough for any chain of N residues.
    """
    c = np.asarray(coordinates)
    hp = np.asarray(hp)
    s, n, _ = c.shape
    size = n if size is None else size
    low, high = c.min(axis=1), c.max(axis=1)
    shift = (size - (high - low + 1)) // 2 - low
    c = c + shift[:, None, :]
    grid = np.zeros((s, 3, size, size), dtype=np.float32)
    rows = np.repeat(np.arange(s), n)
    x, y = c[:, :, 0].ravel(), c[:, :, 1].ravel()
    h = hp.ravel().astype(bool)
    grid[rows[h], 0, y[h], x[h]] = 1.0
    grid[rows[~h], 1, y[~h], x[~h]] = 1.0
    grid[rows, 2, y, x] = np.tile((np.arange(n) + 1) / n, s)
    return grid


# ----------------------------------------------------------------------
# Symmetry augmentation


def mirror(batch: dict, permutation) -> dict:
    """Mirror image of a batch of records: LEFT and RIGHT exchanged in the
    turns, the action slots permuted. Energies and distances do not change.

    A rotation of the lattice changes nothing in the turns or in the actions;
    for the grid encoding use rotate_coordinates.
    """
    permutation = np.asarray(permutation)
    out = dict(batch)
    turns = batch["turns"]
    out["turns"] = np.where(turns == 1, 2, np.where(turns == 2, 1, turns)).astype(turns.dtype)
    # The permutation is an involution: the slot a of the mirror image is the
    # slot permutation[a] of the original.
    for name in ("mask", "next_distance", "next_energy_delta"):
        if name in batch:
            out[name] = batch[name][:, permutation]
    return out


MIRRORED_FIELDS = ("turns", "mask", "next_distance", "next_energy_delta")


def augment(batch: dict, permutation, rng: np.random.Generator, p: float = 0.5) -> dict:
    """Mirrors each record of the batch with probability p (default 0.5).

    The records are stored in canonical form, whose first turn that is not
    FORWARD is always LEFT: without this augmentation the model would never
    see the other half of the states.
    """
    flip = rng.random(len(batch["turns"])) < p
    mirrored = mirror(batch, permutation)
    out = dict(batch)
    for name in MIRRORED_FIELDS:
        if name in batch:
            out[name] = np.where(flip[:, None], mirrored[name], batch[name])
    return out


def rotate_coordinates(coordinates, quarter_turns: int) -> np.ndarray:
    """Coordinates rotated by quarter_turns × 90 degrees."""
    c = np.asarray(coordinates)
    x, y = c[..., 0], c[..., 1]
    for _ in range(quarter_turns % 4):
        x, y = -y, x
    return np.stack([x, y], axis=-1)
