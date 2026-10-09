"""Reading of the configuration file (TOML)."""

import tomllib
from pathlib import Path


def load_config(path: str | Path) -> dict:
    """Reads the TOML file and returns the parameters as a nested dictionary."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    # TODO: validation (required keys, types, length == len(sequence),
    # sequence made only of H and P, chain that fits in the grid).
    return config
