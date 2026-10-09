"""Minimal checks: the C++ extension can be imported and the default configuration can be read."""

from pathlib import Path

from hpfold.config import load_config

ROOT = Path(__file__).resolve().parent.parent


def test_core_extension_is_importable():
    from hpfold import _core

    assert hasattr(_core, "run_remc")


def test_default_config_loads():
    config = load_config(ROOT / "config" / "default.toml")

    assert {"lattice", "protein", "remc", "output"} <= config.keys()
    assert config["protein"]["length"] == len(config["protein"]["sequence"])
