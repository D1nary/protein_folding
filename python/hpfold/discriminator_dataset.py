"""Dataset for the decoy discriminator: native or decoy conformation.

Every example is a pair (HP sequence, conformation) with a binary label
(1 native, 0 decoy) and a continuous stability score (the HP energy and its
gap from the lowest energy of the sequence). The model itself is not built
here.

For every random sequence a REMC run (``[discriminator_dataset.remc]``),
started from the straight chain, provides the states: the trajectories of
the walkers and the snapshots of the replicas.

- Natives (label 1): all the distinct states with the lowest energy reached
  by the REMC, i.e. every degenerate ground state that was found.
- Decoys (label 0): states visited along the run with an energy strictly
  higher than that minimum, chosen in three tiers of difficulty (``near``,
  ``mid``, ``far`` from the native energy); the ``near`` tier prefers kinetic
  traps, local minima of the energy under the move set.

Conformations are identified by their canonical key (canonical.hpp), so
conformations related by a translation, rotation or reflection of the lattice
count once. Each sequence gets as many decoys as natives (balance 1:1 per
sequence), and the train / val / test split is by sequence.

This pipeline is independent of the policy dataset (policy_dataset.py, phase
``policy_dataset``): it only imports some of its helper functions and writes
to its own folder, ``output.discriminator_dataset_dir``.
"""

import csv
import json
from pathlib import Path

import numpy as np

from hpfold import _core
from hpfold.policy_dataset import (
    _git_commit,
    assign_splits,
    candidate_sequences,
    contact_map,
    keys_to_turns,
    straight_chain,
    turns_to_coordinates,
)

FORMAT_VERSION = 1
SPLITS = ("train", "val", "test")
# Difficulty of a decoy (field `tier`); natives have tier "native".
TIERS = ("near", "mid", "far")
# Letters of the field `turns`: FORWARD, LEFT, RIGHT, as in canonical.hpp.
TURN_LETTERS = "FLR"
# Where a state was seen first (field `source`). "remc_best" is the best
# conformation returned by the REMC, added in case the recorded trajectories
# stopped before reaching it.
SOURCES = ("snapshot", "trajectory", "remc_best")

EXAMPLE_FIELDS = (
    "seq_id",
    "sequence",
    "split",
    "label",
    "energy",
    "e_min",
    "gap",
    "tier",
    "kinetic_trap",
    "native_overlap",
    "turns",
    "coordinates",
    "key",
    "source",
    "walker",
    "replica",
    "temperature",
    "cycle",
    "step",
)

SEQUENCE_FIELDS = (
    "seq_id",
    "sequence",
    "split",
    "e_min",
    "exact_e_min",
    "remc_best_energy",
    "natives_found",
    "positives",
    "decoys",
    "visited_states",
)


# ----------------------------------------------------------------------
# Energies and contacts


def _h_contacts(keys, sequence: str, chunk: int = 4096):
    """Yields, chunk by chunk, the H-H contact matrices (S, N, N) of the keys."""
    n = len(sequence)
    h = np.array([c == "H" for c in sequence])
    hh = h[:, None] & h[None, :]
    keys = np.asarray(keys, dtype=np.uint64)
    for a in range(0, len(keys), chunk):
        coordinates = turns_to_coordinates(keys_to_turns(keys[a : a + chunk], n))
        yield contact_map(coordinates) & hh


def energies(keys, sequence: str) -> np.ndarray:
    """HP energy of every canonical key: minus the number of H-H contacts."""
    parts = [-(c.sum(axis=(1, 2)) // 2) for c in _h_contacts(keys, sequence)]
    return np.concatenate(parts).astype(np.int32) if parts else np.zeros(0, dtype=np.int32)


def native_overlap(keys, natives, sequence: str) -> np.ndarray:
    """Fraction of the H-H contacts of a native also present in each state,
    the highest over the given natives (1.0 for a native itself)."""
    native_contacts = np.concatenate(list(_h_contacts(natives, sequence)))
    per_native = native_contacts.sum(axis=(1, 2))
    result = []
    for contacts in _h_contacts(keys, sequence):
        shared = np.einsum("sij,kij->sk", contacts.astype(np.int32),
                           native_contacts.astype(np.int32))
        result.append((shared / np.maximum(per_native, 1)).max(axis=1))
    return np.concatenate(result) if result else np.zeros(0)


def is_kinetic_trap(sequence: str, key: int, energy: int) -> bool:
    """True if no single move (any action slot) lowers the energy of the state:
    a local minimum in which the Monte Carlo search can get stuck."""
    neighbours = [k for _, k in _core.successors(len(sequence), key)]
    return bool(neighbours) and energies(neighbours, sequence).min() >= energy


def tier_of(gap: int, e_min: int, edges) -> int:
    """Index in TIERS of a decoy with the given gap from the lowest energy.

    The gap is measured relative to |e_min| (the largest possible gap, that of
    a state without contacts): up to edges[0] it is near, up to edges[1] mid,
    then far. A gap of 1, the smallest one, is always near.
    """
    fraction = gap / abs(e_min)
    if gap == 1 or fraction <= edges[0]:
        return 0
    return 1 if fraction <= edges[1] else 2


def tier_quotas(count: int, available) -> list[int]:
    """Decoys to take from each tier: `count` split evenly, the remainder to
    the nearest tiers, and what a tier cannot give moved to the others."""
    quotas = [count // len(TIERS) + (t < count % len(TIERS)) for t in range(len(TIERS))]
    quotas = [min(q, a) for q, a in zip(quotas, available)]
    missing = count - sum(quotas)
    for t in range(len(TIERS)):
        extra = min(missing, available[t] - quotas[t])
        quotas[t] += extra
        missing -= extra
    return quotas


# ----------------------------------------------------------------------
# Examples of one sequence


def sample_remc(sequence: str, seq_id: int, config: dict):
    """Runs the REMC of `[discriminator_dataset.remc]`: (recorder, result)."""
    lattice = config["lattice"]
    settings = config["discriminator_dataset"]
    remc = settings["remc"]
    params = _core.RemcParams(
        n_replicas=remc["n_replicas"],
        t_min=remc["t_min"],
        t_max=remc["t_max"],
        local_steps=remc["local_steps"],
        n_cycles=remc["n_cycles"],
        rho=remc["rho"],
        seed=remc["seed"] + seq_id,
    )
    recorder = _core.SampleRecorder(settings["max_recorded_steps"], remc["save_every"])
    initial = straight_chain(sequence, lattice["width"], lattice["height"])
    result = _core.run_remc(initial, params, recorder)
    return recorder, result


def visited_states(recorder, result, temperatures) -> dict:
    """Distinct states visited by the REMC, each with the metadata of the
    first time it was seen: snapshots first (they know the temperature), then
    the trajectories, then the best conformation of the run."""
    snapshots = recorder.snapshots
    trajectories = recorder.trajectories
    m = len(snapshots["key"])
    lengths = [len(t) for t in trajectories]

    keys = np.concatenate(
        [snapshots["key"], *trajectories,
         np.array([_core.canonical_key(result.best)], dtype=np.uint64)]
    )
    source = np.concatenate([
        np.full(m, SOURCES.index("snapshot")),
        np.full(sum(lengths), SOURCES.index("trajectory")),
        [SOURCES.index("remc_best")],
    ])
    walker = np.concatenate(
        [snapshots["walker"], np.repeat(np.arange(len(lengths)), lengths), [-1]]
    )
    replica = np.concatenate([snapshots["replica"], np.full(sum(lengths) + 1, -1)])
    cycle = np.concatenate([snapshots["cycle"], np.full(sum(lengths) + 1, -1)])
    step = np.concatenate(
        [np.full(m, -1), *[np.arange(length) for length in lengths], [-1]]
    )
    temperature = np.full(len(keys), np.nan)
    temperature[:m] = np.asarray(temperatures)[snapshots["replica"]]

    unique, first = np.unique(keys, return_index=True)
    return {
        "key": unique,
        "source": source[first],
        "walker": walker[first],
        "replica": replica[first],
        "temperature": temperature[first],
        "cycle": cycle[first],
        "step": step[first],
    }


def sequence_examples(sequence: str, seq_id: int, config: dict):
    """Natives and decoys of one sequence.

    Returns (examples, info), or (None, reason) if the sequence is dropped:
    lowest energy 0 (every state would be native) or, when the exact check
    applies, a REMC run that did not reach the true ground state (its
    "natives" would be decoys).
    """
    settings = config["discriminator_dataset"]
    lattice = config["lattice"]
    rng = np.random.default_rng([settings["seed"], seq_id])
    n = len(sequence)

    recorder, result = sample_remc(sequence, seq_id, config)
    states = visited_states(recorder, result, result.temperatures)
    energy = energies(states["key"], sequence)
    e_min = int(energy.min())
    if e_min == 0:
        return None, "lowest energy 0"

    exact_e_min = None
    if n <= settings["exact_check_max_length"]:
        exact_e_min = _core.exact_ground_state(sequence, lattice["width"], lattice["height"]).energy
        if e_min > exact_e_min:
            return None, "REMC missed the ground state"

    # Natives: every degenerate state at the lowest energy, up to the cap.
    native_index = np.flatnonzero(energy == e_min)
    count = min(len(native_index), settings["examples_per_class"])
    positives = rng.choice(native_index, size=count, replace=False)

    # Decoys: as many as the natives, spread over the tiers of difficulty.
    decoy_index = np.flatnonzero(energy > e_min)
    gap = energy - e_min
    tiers = np.array([tier_of(int(g), e_min, settings["tier_edges"]) for g in gap[decoy_index]],
                     dtype=np.int64)
    by_tier = [rng.permutation(decoy_index[tiers == t]) for t in range(len(TIERS))]
    quotas = tier_quotas(count, [len(b) for b in by_tier])

    trap = {}

    def check_trap(i):
        if i not in trap:
            trap[i] = is_kinetic_trap(sequence, int(states["key"][i]), int(energy[i]))
        return trap[i]

    # Near tier: kinetic traps first, up to trap_fraction of its quota,
    # looking at a limited number of candidates.
    near = by_tier[0]
    wanted = int(np.ceil(settings["trap_fraction"] * quotas[0]))
    traps = [i for i in near[: settings["trap_search_limit"]] if check_trap(i)][:wanted]
    trap_set = set(traps)
    others = [i for i in near if i not in trap_set]
    chosen = traps + others[: quotas[0] - len(traps)]
    for t in (1, 2):
        chosen += list(by_tier[t][: quotas[t]])
    negatives = np.array(chosen, dtype=np.int64)

    index = np.concatenate([positives, negatives]).astype(np.int64)
    label = np.concatenate([np.ones(len(positives), int), np.zeros(len(negatives), int)])
    keys = states["key"][index]
    turns = keys_to_turns(keys, n)
    coordinates = turns_to_coordinates(turns)
    overlap = native_overlap(keys, states["key"][positives], sequence)

    examples = []
    for row, i in enumerate(index):
        g = int(gap[i])
        examples.append({
            "seq_id": seq_id,
            "sequence": sequence,
            "label": int(label[row]),
            "energy": int(energy[i]),
            "e_min": e_min,
            "gap": g,
            "tier": "native" if g == 0 else TIERS[tier_of(g, e_min, settings["tier_edges"])],
            "kinetic_trap": int(g > 0 and check_trap(int(i))),
            "native_overlap": round(float(overlap[row]), 4),
            "turns": "".join(TURN_LETTERS[t] for t in turns[row]),
            "coordinates": json.dumps(coordinates[row].tolist(), separators=(",", ":")),
            "key": int(keys[row]),
            "source": SOURCES[states["source"][i]],
            "walker": int(states["walker"][i]),
            "replica": int(states["replica"][i]),
            "temperature": "" if np.isnan(states["temperature"][i])
            else round(float(states["temperature"][i]), 6),
            "cycle": int(states["cycle"][i]),
            "step": int(states["step"][i]),
        })
    info = {
        "seq_id": seq_id,
        "sequence": sequence,
        "e_min": e_min,
        "exact_e_min": "" if exact_e_min is None else exact_e_min,
        "remc_best_energy": result.best_energy,
        "natives_found": len(native_index),
        "positives": len(positives),
        "decoys": len(negatives),
        "visited_states": len(states["key"]),
    }
    return examples, info


# ----------------------------------------------------------------------
# Building and report


def _write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(examples: list[dict], infos: list[dict], dropped: dict) -> dict:
    """Counts for the report: examples per class and split, decoy energies."""
    decoys = [e for e in examples if e["label"] == 0]
    gaps = np.bincount([e["gap"] for e in decoys]) if decoys else np.zeros(0, int)
    return {
        "examples": {
            split: {
                "native": sum(1 for e in examples if e["split"] == split and e["label"] == 1),
                "decoy": sum(1 for e in examples if e["split"] == split and e["label"] == 0),
            }
            for split in SPLITS
        },
        "sequences": {split: sum(1 for i in infos if i["split"] == split) for split in SPLITS},
        "dropped_sequences": dropped,
        "decoy_gap": {str(g): int(c) for g, c in enumerate(gaps) if c},
        "decoy_energy": {
            str(v): sum(1 for e in decoys if e["energy"] == v)
            for v in sorted({e["energy"] for e in decoys})
        },
        "decoy_tier": {t: sum(1 for e in decoys if e["tier"] == t) for t in TIERS},
        "kinetic_traps": sum(e["kinetic_trap"] for e in decoys),
    }


def format_report(s: dict) -> str:
    lines = ["Examples per class (native / decoy):"]
    for split, c in s["examples"].items():
        lines.append(f"  {split:<6} {c['native']:6d} / {c['decoy']:6d}")
    total_native = sum(c["native"] for c in s["examples"].values())
    total_decoy = sum(c["decoy"] for c in s["examples"].values())
    lines.append(f"  {'total':<6} {total_native:6d} / {total_decoy:6d}")
    lines.append("Sequences per split: " + ", ".join(f"{k} {v}" for k, v in s["sequences"].items()))
    if s["dropped_sequences"]:
        lines.append("Dropped sequences: "
                     + ", ".join(f"{k} {v}" for k, v in s["dropped_sequences"].items()))
    lines.append("Decoy tiers: " + ", ".join(f"{k} {v}" for k, v in s["decoy_tier"].items())
                 + f"  (kinetic traps {s['kinetic_traps']})")
    lines.append("Decoy gap from the lowest energy:")
    largest = max(s["decoy_gap"].values(), default=1)
    for g, c in s["decoy_gap"].items():
        lines.append(f"  +{g:<3} {'#' * max(1, round(40 * c / largest)):<40} {c}")
    lines.append("Decoy energy: " + ", ".join(f"{k}: {v}" for k, v in s["decoy_energy"].items()))
    return "\n".join(lines)


def build_dataset(config: dict, verbose: bool = True) -> Path:
    """Generates the dataset in `output.discriminator_dataset_dir`."""
    settings = config["discriminator_dataset"]
    n = settings["length"]
    if n < 4 or n > _core.MAX_KEY_LENGTH:
        raise ValueError(f"discriminator_dataset.length must be between 4 and "
                         f"{_core.MAX_KEY_LENGTH}")

    rng = np.random.default_rng(settings["seed"])
    stream = candidate_sequences(n, settings["h_fraction"], rng)
    examples, infos, dropped = [], [], {}
    attempts = 0
    while len(infos) < settings["n_sequences"]:
        attempts += 1
        if attempts > 20 * settings["n_sequences"]:
            raise RuntimeError("too many sequences dropped: see the report of the dropped ones")
        sequence = next(stream)
        part, info = sequence_examples(sequence, len(infos), config)
        if part is None:
            dropped[info] = dropped.get(info, 0) + 1
            continue
        examples += part
        infos.append(info)
        if verbose:
            print(f"  [{len(infos)}/{settings['n_sequences']}] {sequence}  E_min={info['e_min']}"
                  f"  natives found={info['natives_found']}  examples={2 * info['positives']}")

    split = assign_splits(len(infos), settings["split"], rng)
    for info, s in zip(infos, split):
        info["split"] = SPLITS[s]
    for e in examples:
        e["split"] = infos[e["seq_id"]]["split"]

    out = Path(config["output"]["discriminator_dataset_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        _write_csv(out / f"{name}.csv", EXAMPLE_FIELDS, [e for e in examples if e["split"] == name])
    _write_csv(out / "sequences.csv", SEQUENCE_FIELDS, infos)

    report = summary(examples, infos, dropped)
    (out / "report.txt").write_text(format_report(report) + "\n")
    meta = {
        "format_version": FORMAT_VERSION,
        "length": n,
        "splits": list(SPLITS),
        "tiers": list(TIERS),
        "sources": list(SOURCES),
        "turn_letters": {"F": "FORWARD", "L": "LEFT", "R": "RIGHT"},
        "coordinates": "canonical form: residue 0 at the origin, first bond along +x",
        "summary": report,
        "config": {
            "discriminator_dataset": settings,
            "lattice": config["lattice"],
        },
        "git_commit": _git_commit(),
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out


def run(config: dict) -> None:
    """Builds the dataset and prints the report."""
    out = build_dataset(config)
    print((out / "report.txt").read_text(), end="")
    print(f"Saved to: {out}")


# ----------------------------------------------------------------------
# Reading


def load_meta(directory) -> dict:
    with open(Path(directory) / "meta.json") as f:
        return json.load(f)


def load_split(directory, split: str) -> dict:
    """Examples of one split as numpy arrays.

    Besides the columns of the CSV (strings and numbers), it adds `hp` (S, N)
    with 1 for H, `turns` as codes (S, N-2) 0 forward / 1 left / 2 right and
    `coordinates` as an array (S, N, 2).
    """
    with open(Path(directory) / f"{split}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    n = load_meta(directory)["length"]
    data = {}
    for name in ("seq_id", "label", "energy", "e_min", "gap", "kinetic_trap", "walker",
                 "replica", "cycle", "step"):
        data[name] = np.array([int(r[name]) for r in rows], dtype=np.int32)
    data["native_overlap"] = np.array([float(r["native_overlap"]) for r in rows])
    data["temperature"] = np.array(
        [float(r["temperature"]) if r["temperature"] else np.nan for r in rows]
    )
    data["key"] = np.array([int(r["key"]) for r in rows], dtype=np.uint64)
    for name in ("sequence", "tier", "source"):
        data[name] = np.array([r[name] for r in rows])
    data["hp"] = np.array([[c == "H" for c in r["sequence"]] for r in rows],
                          dtype=np.uint8).reshape(len(rows), n)
    data["turns"] = np.array([[TURN_LETTERS.index(c) for c in r["turns"]] for r in rows],
                             dtype=np.uint8).reshape(len(rows), n - 2)
    data["coordinates"] = np.array([json.loads(r["coordinates"]) for r in rows],
                                   dtype=np.int32).reshape(len(rows), n, 2)
    return data
