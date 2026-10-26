"""Judgement of one conformation: input file, checks and outputs (no window)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tkinter")

from hpfold import __main__ as cli  # noqa: E402
from hpfold import discriminator_model as dm  # noqa: E402
from hpfold import discriminator_score as ds  # noqa: E402
from hpfold.config import load_config  # noqa: E402

# A U of 4 residues: the two ends are in contact.
SEQUENCE = "HPPH"
COORDINATES = [(0, 0), (1, 0), (1, 1), (0, 1)]


def write_input(path, sequence, coordinates):
    pairs = ", ".join(f"[{x}, {y}]" for x, y in coordinates)
    path.write_text(f'sequence = "{sequence}"\ncoordinates = [{pairs}]\n')
    return path


def test_load_input(tmp_path):
    path = write_input(tmp_path / "input.toml", SEQUENCE, COORDINATES)
    assert ds.load_input(path, 4) == (SEQUENCE, COORDINATES)


def test_default_input_file_fits_the_default_dataset():
    config = load_config("config/default.toml")
    length = config["discriminator_dataset"]["length"]
    sequence, coordinates = ds.load_input(config["discriminator_model"]["input_conformation"],
                                          length)
    assert len(sequence) == len(coordinates) == length


def test_coordinates_can_be_anywhere():
    ds.check_input(SEQUENCE, [(x - 5, y - 7) for x, y in COORDINATES], 4)


@pytest.mark.parametrize("sequence, coordinates", [
    ("HPPHH", COORDINATES + [(0, 2)]),             # wrong length
    ("HPPX", COORDINATES),                          # not H or P
    (SEQUENCE, [(0, 0), (1, 0), (2, 0), (2, 2)]),   # not adjacent
    (SEQUENCE, [(0, 0), (1, 0), (0, 0), (0, 1)]),   # cell used twice
    (SEQUENCE, COORDINATES[:3]),                    # a coordinate missing
])
def test_invalid_input(sequence, coordinates):
    with pytest.raises(ValueError):
        ds.check_input(sequence, coordinates, 4)


def test_evaluate():
    torch.manual_seed(0)
    model = dm.Discriminator(length=4, hidden=8, gnn_layers=1, cnn_layers=1, cnn_kernel=3)
    result = ds.evaluate(model, SEQUENCE, COORDINATES)
    assert set(result) == {"prob", "logit", "energy", "e_min", "gap", "exact_energy"}
    assert 0 <= result["prob"] <= 1
    assert result["exact_energy"] == -1  # the H at the two ends
    text = ds.format_result(SEQUENCE, result)
    assert SEQUENCE in text and "Probability of native" in text


def test_cli_phase():
    assert cli.PHASES["discriminator_score"] is cli.discriminator_score
