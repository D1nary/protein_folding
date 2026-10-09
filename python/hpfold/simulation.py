"""Phase 1 — search for the native state with REMC.

Python wrapper around the C++ extension ``hpfold._core``: it translates the
configuration dictionary into the simulation parameters and starts it.
"""


def run(config: dict) -> None:
    """Starts the REMC simulation from the initial configuration."""
    # TODO(phase 1): build the parameters from `config` and call _core.run_remc.
    raise NotImplementedError("Fase 1 (REMC) non ancora implementata")
