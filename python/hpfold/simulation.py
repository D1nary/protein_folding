"""Phase 1 — search for the native state with REMC.

Python wrapper around the C++ extension ``hpfold._core``: it translates the
configuration dictionary into the simulation parameters, starts the simulation
and saves the lowest-energy conformation found.
"""

from hpfold import _core
from hpfold.conformation import load_conformation, save_conformation


def simulate(config: dict) -> _core.RemcResult:
    """Runs the REMC from the initial conformation and returns the result.

    Sequence and coordinates come from the file written in phase 0
    (``protein.initial_conformation``), not from ``protein.sequence``.
    """
    lattice = config["lattice"]
    remc = config["remc"]

    sequence, coordinates = load_conformation(config["protein"]["initial_conformation"])
    # The constructor checks that the chain is a self-avoiding walk inside the
    # lattice and raises ValueError otherwise.
    initial = _core.Conformation(sequence, coordinates, lattice["width"], lattice["height"])

    params = _core.RemcParams(
        n_replicas=remc["n_replicas"],
        t_min=remc["t_min"],
        t_max=remc["t_max"],
        local_steps=remc["local_steps"],
        n_cycles=remc["n_cycles"],
        rho=remc["rho"],
        seed=remc["seed"],
    )
    return _core.run_remc(initial, params)


def run(config: dict) -> None:
    """Starts the REMC simulation and saves the best conformation found."""
    result = simulate(config)

    path = config["output"]["native_conformation"]
    save_conformation(
        path, result.best.sequence, result.best.coordinates, energy=result.best_energy
    )

    print(f"Sequence:       {result.best.sequence}")
    print(f"Minimum energy: {result.best_energy}")
    print(f"Swaps accepted: {result.swap_accepted}/{result.swap_attempts}")
    print(f"Saved to:       {path}")
