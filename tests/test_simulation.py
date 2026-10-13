"""Phase 1: REMC parameters, reproducibility and output file."""

import json

import pytest

from hpfold import _core, simulation, verification
from hpfold.conformation import load_conformation, save_conformation, validate_conformation

SEQUENCE = "HPHPPHHPHPPHPHHH"
# Exact ground-state energy of SEQUENCE, from an exhaustive enumeration of the
# self-avoiding walks of 16 residues.
GROUND_STATE_ENERGY = -7
WIDTH = HEIGHT = 30


def make_params(**overrides):
    values = dict(
        n_replicas=5, t_min=0.160, t_max=0.220, local_steps=500, n_cycles=200, rho=0.4, seed=42
    )
    values.update(overrides)
    return _core.RemcParams(**values)


def initial_conformation():
    coordinates = [(7 + i, 15) for i in range(len(SEQUENCE))]
    return _core.Conformation(SEQUENCE, coordinates, WIDTH, HEIGHT)


def test_temperatures_are_linearly_spaced():
    assert _core.temperature_ladder(5, 0.160, 0.220) == pytest.approx(
        [0.160, 0.175, 0.190, 0.205, 0.220]
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_replicas": 1},
        {"t_min": 0.0},
        {"t_min": 0.3},  # above t_max
        {"local_steps": 0},
        {"n_cycles": 0},
        {"rho": 1.5},
    ],
)
def test_invalid_parameters_are_rejected(overrides):
    with pytest.raises(ValueError):
        _core.run_remc(initial_conformation(), make_params(**overrides))


def test_same_seed_gives_the_same_run():
    first = _core.run_remc(initial_conformation(), make_params(seed=7))
    second = _core.run_remc(initial_conformation(), make_params(seed=7))

    assert first.best_energy == second.best_energy
    assert first.best.coordinates == second.best.coordinates
    assert first.final_energies == second.final_energies
    assert first.swap_accepted == second.swap_accepted


def test_different_seeds_give_different_runs():
    first = _core.run_remc(initial_conformation(), make_params(seed=1, n_cycles=20))
    second = _core.run_remc(initial_conformation(), make_params(seed=2, n_cycles=20))
    assert first.best.coordinates != second.best.coordinates


def test_best_conformation_is_valid_and_consistent():
    initial = initial_conformation()
    result = _core.run_remc(initial, make_params())

    validate_conformation(SEQUENCE, result.best.coordinates, WIDTH, HEIGHT)
    assert result.best.sequence == SEQUENCE
    assert result.best_energy == result.best.energy()
    # The best energy can only improve on the replicas and on the start.
    assert result.best_energy <= min(result.final_energies) <= initial.energy()
    assert initial.coordinates == [(7 + i, 15) for i in range(len(SEQUENCE))]  # input untouched
    assert len(result.final_energies) == 5
    # 5 replicas: two pairs are proposed in every one of the 200 cycles.
    assert 0 <= result.swap_accepted <= result.swap_attempts == 400


@pytest.mark.parametrize("rho", [0.0, 0.4, 1.0])
def test_remc_never_goes_below_the_ground_state(rho):
    result = _core.run_remc(initial_conformation(), make_params(rho=rho))
    assert result.best_energy >= GROUND_STATE_ENERGY


def test_remc_finds_the_ground_state():
    result = _core.run_remc(initial_conformation(), make_params(n_cycles=1000))
    assert result.best_energy == GROUND_STATE_ENERGY


def test_exact_enumeration_on_known_sequences():
    # Figure 1 of Thachuk et al. (2007) and two standard 2D benchmarks.
    assert _core.exact_ground_state("HPPH", WIDTH, HEIGHT).energy == -1
    assert _core.exact_ground_state(SEQUENCE, WIDTH, HEIGHT).energy == GROUND_STATE_ENERGY
    assert _core.exact_ground_state("HPHPPHHPHPPHPHHPPHPH", WIDTH, HEIGHT).energy == -9
    # No H-H contact is possible without H residues, or with a single one.
    assert _core.exact_ground_state("PPPPPP", WIDTH, HEIGHT).energy == 0
    assert _core.exact_ground_state("H", WIDTH, HEIGHT).energy == 0


def test_exact_conformation_is_valid_and_consistent():
    result = _core.exact_ground_state(SEQUENCE, WIDTH, HEIGHT)
    validate_conformation(SEQUENCE, result.best.coordinates, WIDTH, HEIGHT)
    assert result.best.energy() == result.energy


def test_exact_enumeration_respects_the_lattice():
    # In a 4x1 lattice the only conformation is the straight chain; a 2x2
    # square lets the two ends touch.
    assert _core.exact_ground_state("HPPH", 4, 1).energy == 0
    assert _core.exact_ground_state("HPPH", 1, 4).energy == 0
    assert _core.exact_ground_state("HPPH", 2, 2).energy == -1
    # A wide and low lattice: the walk is rotated to fit.
    result = _core.exact_ground_state(SEQUENCE, 8, 3)
    validate_conformation(SEQUENCE, result.best.coordinates, 8, 3)
    assert result.energy >= GROUND_STATE_ENERGY

    with pytest.raises(ValueError):
        _core.exact_ground_state("HPPH", 3, 1)  # the chain does not fit
    with pytest.raises(ValueError):
        _core.exact_ground_state("HXPH", WIDTH, HEIGHT)


def test_run_saves_the_best_conformation(tmp_path, capsys):
    initial_path = tmp_path / "initial.json"
    output_path = tmp_path / "out" / "native.json"
    save_conformation(initial_path, SEQUENCE, initial_conformation().coordinates)
    config = {
        "lattice": {"width": WIDTH, "height": HEIGHT},
        # Different from the file on purpose: phase 1 must read only the file.
        "protein": {"sequence": "P" * 16, "initial_conformation": str(initial_path)},
        "remc": {
            "n_replicas": 5, "t_min": 0.160, "t_max": 0.220, "local_steps": 500,
            "n_cycles": 50, "rho": 0.4, "seed": 42,
            "save_every": 10,  # not used in phase 1
        },
        "output": {"native_conformation": str(output_path)},
    }  # fmt: skip

    simulation.run(config)

    sequence, coordinates = load_conformation(output_path)
    assert sequence == SEQUENCE
    validate_conformation(sequence, coordinates, WIDTH, HEIGHT)

    energy = json.loads(output_path.read_text())["energy"]
    assert energy == _core.Conformation(sequence, coordinates, WIDTH, HEIGHT).energy() < 0
    output = capsys.readouterr().out
    assert f"Minimum energy: {energy}" in output
    # The exact enumeration is a separate phase: the simulation does not run it.
    assert "Exact minimum" not in output


def verification_config(tmp_path, coordinates, max_length=16):
    path = tmp_path / "native.json"
    save_conformation(path, SEQUENCE, coordinates)
    return {
        "lattice": {"width": WIDTH, "height": HEIGHT},
        "exact": {"max_length": max_length},
        "output": {"native_conformation": str(path)},
    }


def test_verify_recognises_the_ground_state(tmp_path, capsys):
    ground_state = _core.exact_ground_state(SEQUENCE, WIDTH, HEIGHT).best
    verification.run(verification_config(tmp_path, ground_state.coordinates))

    output = capsys.readouterr().out
    assert f"REMC energy:    {GROUND_STATE_ENERGY}" in output
    assert f"Exact minimum:  {GROUND_STATE_ENERGY} -> the REMC found the ground state" in output


def test_verify_reports_a_conformation_above_the_ground_state(tmp_path, capsys):
    # The straight chain has no contacts.
    verification.run(verification_config(tmp_path, initial_conformation().coordinates))

    output = capsys.readouterr().out
    assert "REMC energy:    0" in output
    assert f"Exact minimum:  {GROUND_STATE_ENERGY} -> the REMC did NOT reach" in output


def test_verify_skips_chains_longer_than_max_length(tmp_path, capsys):
    config = verification_config(tmp_path, initial_conformation().coordinates, max_length=15)
    verification.run(config)
    assert "Exact minimum:  not computed" in capsys.readouterr().out
