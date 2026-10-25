"""Dataset for the decoy discriminator: labels, balance, deduplication,
splits and files."""

import copy
import csv
import json

import numpy as np
import pytest

from hpfold import __main__ as cli
from hpfold import _core
from hpfold import discriminator_dataset as dd
from hpfold import policy_dataset
from hpfold.config import load_config

# ----------------------------------------------------------------------
# Helpers


def test_energies_match_the_core():
    sequence = "HPHPPHHPHPPHPHHH"
    n = len(sequence)
    rng = np.random.default_rng(0)
    keys = [0]
    for _ in range(300):
        options = _core.successors(n, keys[-1])
        keys.append(options[rng.integers(len(options))][1])
    expected = [_core.conformation_from_key(sequence, k).energy() for k in keys]
    assert dd.energies(keys, sequence).tolist() == expected


def test_kinetic_trap_is_a_local_minimum():
    sequence = "HPHPPHHPHPPHPHHH"
    # The straight chain has energy 0 and can always fold a corner: not a trap.
    assert not dd.is_kinetic_trap(sequence, 0, 0)
    ground = _core.exact_ground_state(sequence, 30, 30)
    key = _core.canonical_key(ground.best)
    assert dd.is_kinetic_trap(sequence, key, ground.energy)


@pytest.mark.parametrize("gap, e_min, tier", [(1, -9, 0), (3, -9, 0), (4, -9, 1), (6, -9, 1),
                                               (7, -9, 2), (1, -2, 0), (2, -2, 2)])
def test_tier_of(gap, e_min, tier):
    assert dd.tier_of(gap, e_min, [0.34, 0.67]) == tier


def test_tier_quotas():
    assert dd.tier_quotas(7, [10, 10, 10]) == [3, 2, 2]
    assert dd.tier_quotas(7, [1, 10, 0]) == [1, 6, 0]
    assert dd.tier_quotas(5, [2, 1, 1]) == [2, 1, 1]  # not enough decoys


# ----------------------------------------------------------------------
# Dataset files


def small_config(directory):
    config = copy.deepcopy(load_config("config/default.toml"))
    config["lattice"] = {"width": 20, "height": 20}
    config["output"]["discriminator_dataset_dir"] = str(directory)
    d = config["discriminator_dataset"]
    d.update(n_sequences=6, length=10, examples_per_class=5, max_recorded_steps=20000)
    d["remc"].update(n_replicas=3, local_steps=50, n_cycles=60, save_every=4)
    return config


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    directory = tmp_path_factory.mktemp("discriminator_dataset")
    config = small_config(directory)
    dd.build_dataset(config, verbose=False)
    return directory, config


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def all_examples(directory):
    return [r for s in dd.SPLITS for r in read_rows(directory / f"{s}.csv")]


def test_files(built):
    directory, _ = built
    for name in [f"{s}.csv" for s in dd.SPLITS] + ["sequences.csv", "provenance.csv",
                                                    "meta.json", "report.txt"]:
        assert (directory / name).exists()
    assert tuple(read_rows(directory / "train.csv")[0]) == dd.EXAMPLE_FIELDS
    assert tuple(read_rows(directory / "provenance.csv")[0]) == dd.PROVENANCE_FIELDS
    assert json.loads((directory / "meta.json").read_text())["length"] == 10


def test_provenance_is_separate(built):
    directory, _ = built
    # The provenance fields predict the label: none of them is in the split files.
    shared = set(dd.EXAMPLE_FIELDS) & set(dd.PROVENANCE_FIELDS)
    assert shared == {"seq_id", "key", "split"}
    provenance = read_rows(directory / "provenance.csv")
    examples = all_examples(directory)
    assert [(p["seq_id"], p["key"], p["split"]) for p in provenance] == \
        [(e["seq_id"], e["key"], e["split"]) for e in examples]
    assert {p["source"] for p in provenance} <= set(dd.SOURCES)
    for split in dd.SPLITS:
        data = dd.load_split(directory, split)
        origin = dd.load_provenance(directory, split)
        assert (origin["key"] == data["key"]).all()
        assert (origin["seq_id"] == data["seq_id"]).all()


def test_labels_and_energies(built):
    directory, _ = built
    for r in all_examples(directory):
        sequence, energy, e_min = r["sequence"], int(r["energy"]), int(r["e_min"])
        # Canonical coordinates may be negative: shift them inside the lattice.
        coordinates = np.array(json.loads(r["coordinates"]))
        shifted = [tuple(map(int, c)) for c in coordinates - coordinates.min(axis=0)]
        conformation = _core.Conformation(sequence, shifted, 20, 20)
        assert conformation.energy() == energy
        assert _core.canonical_key(conformation) == int(r["key"])
        assert int(r["gap"]) == energy - e_min
        if r["label"] == "1":
            assert energy == e_min and r["tier"] == "native"
        else:
            assert energy > e_min and r["tier"] in dd.TIERS


def test_balance_and_no_duplicates(built):
    directory, _ = built
    rows = all_examples(directory)
    sequences = read_rows(directory / "sequences.csv")
    for info in sequences:
        mine = [r for r in rows if r["seq_id"] == info["seq_id"]]
        positives = sum(r["label"] == "1" for r in mine)
        assert positives == sum(r["label"] == "0" for r in mine) == int(info["positives"])
        assert positives == min(int(info["natives_found"]), 5)
        keys = [r["key"] for r in mine]
        assert len(keys) == len(set(keys))  # also covers rotations and reflections
        if info["exact_e_min"]:
            assert info["e_min"] == info["exact_e_min"]


def test_split_by_sequence(built):
    directory, _ = built
    seen = {}
    for split in dd.SPLITS:
        for r in read_rows(directory / f"{split}.csv"):
            assert r["split"] == split
            assert seen.setdefault(r["sequence"], split) == split
    sequences = read_rows(directory / "sequences.csv")
    assert {s["split"] for s in sequences} == set(dd.SPLITS)
    groups = [min(s["sequence"], s["sequence"][::-1]) for s in sequences]
    assert len(groups) == len(set(groups))


def test_load_split(built):
    directory, _ = built
    data = dd.load_split(directory, "train")
    rows = read_rows(directory / "train.csv")
    assert data["coordinates"].shape == (len(rows), 10, 2)
    assert data["turns"].shape == (len(rows), 8)
    assert data["hp"].shape == (len(rows), 10)
    assert set(np.unique(data["label"])) <= {0, 1}
    assert not (set(dd.PROVENANCE_FIELDS) - {"seq_id", "key", "split"}) & set(data)


def test_model_inputs_are_the_whitelist(built):
    directory, _ = built
    inputs = dd.model_inputs(dd.load_split(directory, "train"))
    assert tuple(inputs) == dd.MODEL_INPUTS == ("hp", "turns", "coordinates")


def test_reproducible(built, tmp_path):
    directory, config = built
    config = copy.deepcopy(config)
    config["output"]["discriminator_dataset_dir"] = str(tmp_path)
    dd.build_dataset(config, verbose=False)
    for name in [f"{s}.csv" for s in dd.SPLITS] + ["provenance.csv"]:
        assert (tmp_path / name).read_text() == (directory / name).read_text()


def test_cli_phases():
    assert cli.PHASES["discriminator_dataset"] is dd.run
    assert cli.PHASES["policy_dataset"] is policy_dataset.run
    assert "dataset" not in cli.PHASES


def test_default_output_folders_are_separate():
    output = load_config("config/default.toml")["output"]
    assert output["policy_dataset_dir"].endswith("policy_dataset")
    assert output["discriminator_dataset_dir"].endswith("discriminator_dataset")
