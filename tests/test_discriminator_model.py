"""Decoy discriminator: inputs, symmetries, metrics, training and files."""

import copy
import csv
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hpfold import __main__ as cli  # noqa: E402
from hpfold import discriminator_dataset as dd  # noqa: E402
from hpfold import discriminator_model as dm  # noqa: E402
from hpfold.config import load_config  # noqa: E402
from hpfold.policy_dataset import contact_map, rotate_coordinates  # noqa: E402

# ----------------------------------------------------------------------
# Small dataset and model, shared by the tests


def small_config(directory):
    config = copy.deepcopy(load_config("config/default.toml"))
    config["lattice"] = {"width": 20, "height": 20}
    config["output"]["discriminator_dataset_dir"] = str(directory / "dataset")
    config["output"]["discriminator_model_dir"] = str(directory / "model")
    d = config["discriminator_dataset"]
    d.update(n_sequences=8, length=10, examples_per_class=5, max_recorded_steps=20000)
    d["remc"].update(n_replicas=3, local_steps=50, n_cycles=60, save_every=4)
    config["discriminator_model"].update(hidden=16, gnn_layers=2, cnn_layers=2, max_epochs=3,
                                         batch_size=16)
    return config


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("discriminator_model")
    config = small_config(directory)
    dd.build_dataset(config, verbose=False)
    out = dm.run(config, verbose=False)
    return config, out


def examples(config, split="test"):
    return dm.load_examples(config["output"]["discriminator_dataset_dir"], split)


# ----------------------------------------------------------------------
# Inputs


def test_node_features():
    hp = torch.tensor([[1.0, 0.0, 0.0, 1.0, 1.0]])
    x = dm.node_features(hp)
    assert x.shape == (1, 5, dm.N_NODE_FEATURES)
    assert x[0, :, 0].tolist() == hp[0].tolist()
    assert x[0, :, 1].tolist() == [0, 1, 0, 1, 0]  # parity
    assert x[0, :, 2].tolist() == [0, 0.25, 0.5, 0.75, 1]  # position
    assert x[0, :, 3].tolist() == [1, 0, 0, 0, 1]  # chain end


def test_sequence_features():
    hp = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0, 0.0]])  # H at 0, 1, 3
    # total, even, odd, min(even, odd), at the ends
    assert dm.sequence_features(hp)[0].tolist() == [3, 1, 2, 1, 1]


def test_energy_from_the_inputs_matches_the_dataset(trained):
    config, _ = trained
    for split in dd.SPLITS:
        e = examples(config, split)
        assert torch.equal(dm.hh_energy(e["hp"], e["contacts"]), e["energy"])


def test_the_model_never_sees_the_answer(trained):
    config, _ = trained
    e = examples(config)
    # Only the sequence and the contact map reach the models.
    assert set(e) - {"info"} == {"hp", "contacts", *dm.TARGETS}
    assert set(dd.MODEL_INPUTS) == {"hp", "turns", "coordinates"}


def test_invariant_to_rotations_and_reflections(trained):
    config, out = trained
    data = dd.load_split(config["output"]["discriminator_dataset_dir"], "test")
    model = dm.load_model(out)
    hp = torch.tensor(data["hp"], dtype=torch.float32)
    expected = None
    for quarter_turns in range(4):
        for mirror in (1, -1):
            c = rotate_coordinates(data["coordinates"], quarter_turns) * np.array([1, mirror])
            contacts = torch.tensor(contact_map(c), dtype=torch.float32)
            prob = dm.predict(model, hp, contacts)["prob"]
            if expected is None:
                expected = prob
            np.testing.assert_array_equal(prob, expected)


def test_reverse_chain_is_the_chain_read_backwards(trained):
    config, _ = trained
    data = dd.load_split(config["output"]["discriminator_dataset_dir"], "train")
    hp = torch.tensor(data["hp"], dtype=torch.float32)
    contacts = torch.tensor(contact_map(data["coordinates"]), dtype=torch.float32)
    hp_r, contacts_r = dm.reverse_chain(hp, contacts)
    assert torch.equal(hp_r, torch.tensor(data["hp"][:, ::-1].copy(), dtype=torch.float32))
    reversed_map = contact_map(data["coordinates"][:, ::-1])
    assert torch.equal(contacts_r, torch.tensor(reversed_map, dtype=torch.float32))
    # Same physical state: same energy.
    assert torch.equal(dm.hh_energy(hp_r, contacts_r), dm.hh_energy(hp, contacts))


# ----------------------------------------------------------------------
# Metrics


def brute_force_auroc(score, label):
    pos = [s for s, y in zip(score, label) if y]
    neg = [s for s, y in zip(score, label) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_auroc():
    assert dm.auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert dm.auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert dm.auroc([0.5, 0.5, 0.5], [1, 0, 1]) == 0.5
    assert np.isnan(dm.auroc([0.1, 0.2], [1, 1]))
    rng = np.random.default_rng(0)
    score = rng.integers(0, 5, 200) / 4  # many ties
    label = rng.integers(0, 2, 200)
    assert dm.auroc(score, label) == pytest.approx(brute_force_auroc(score, label))


# ----------------------------------------------------------------------
# Training and files


def test_files_and_metrics(trained):
    config, out = trained
    for name in ("discriminator.pt", "metrics.json", "history.csv", "test_predictions.csv",
                 "report.txt"):
        assert (out / name).exists(), name
    metrics = json.loads((out / "metrics.json").read_text())["metrics"]
    assert list(metrics) == [dm.MAIN_MODEL] + list(config["discriminator_model"]["baselines"])
    for split in ("val", "test"):
        m = metrics[dm.MAIN_MODEL][split]
        assert 0 <= m["accuracy"] <= 1 and 0 <= m["auroc"] <= 1
        assert {"mae_energy", "mae_e_min", "mae_gap", "accuracy_by_tier"} <= set(m)
        assert "mae_energy" not in metrics["B0a"][split]


def test_saved_model_gives_the_saved_predictions(trained):
    config, out = trained
    model = dm.load_model(out)
    with open(out / "test_predictions.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    e = examples(config)
    prob = dm.predict(model, e["hp"], e["contacts"])["prob"]
    assert len(rows) == len(prob)
    np.testing.assert_allclose([float(r[f"prob_{dm.MAIN_MODEL}"]) for r in rows], prob,
                               atol=1e-5)
    # The same through score(), from sequences and coordinates.
    data = dd.load_split(config["output"]["discriminator_dataset_dir"], "test")
    np.testing.assert_allclose(dm.score(model, data["sequence"], data["coordinates"]), prob,
                               rtol=1e-6)


def test_training_is_reproducible(trained, tmp_path):
    config, out = trained
    again = copy.deepcopy(config)
    again["output"]["discriminator_model_dir"] = str(tmp_path / "model")
    dm.run(again, verbose=False)
    first = json.loads((out / "metrics.json").read_text())["metrics"]
    second = json.loads((tmp_path / "model" / "metrics.json").read_text())["metrics"]
    assert first == second


def test_train_sequences_subset(trained):
    config, _ = trained
    directory = config["output"]["discriminator_dataset_dir"]
    full = dm.load_examples(directory, "train")
    sub = dm.load_examples(directory, "train", train_sequences=2, seed=0)
    assert len(np.unique(sub["info"]["seq_id"])) == 2
    assert set(sub["info"]["seq_id"]) <= set(full["info"]["seq_id"])


def test_unknown_model():
    with pytest.raises(ValueError):
        dm.make_model("B9", 10, load_config("config/default.toml")["discriminator_model"])


def test_cli_phase():
    assert "discriminator_train" in cli.PHASES
    assert load_config("config/default.toml")["output"]["discriminator_model_dir"]
