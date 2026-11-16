# `python/hpfold/`

Python package `hpfold`: command line, pipeline orchestration, datasets, ML model and GUI. The heavy work is delegated to the C++ core through the compiled module `hpfold._core`. Each phase has a module with a `run(config)` function as its entry point.

| File | Phase | Role |
|---|---|---|
| `__main__.py` | — | CLI `python -m hpfold --config <file> <phase>`: maps each phase name to its `run`. |
| `config.py` | — | Reading of the TOML configuration (`load_config`). |
| `discriminator_dataset.py` | `discriminator_dataset` | Dataset of natives and decoys: REMC per sequence, `near`/`mid`/`far` tiers, kinetic traps, 1:1 balancing, split by sequence, CSV and report. |
| `discriminator_model.py` | `discriminator_train` | PyTorch model (GNN on contacts + 1D CNN on the sequence, auxiliary heads for energy, `E_min` and gap), `B0a`/`B0b`/`B1b` baselines, training with early stopping, metrics, `load_model` and `score`. |
| `discriminator_score.py` | `discriminator_score` | Judgment of a conformation drawn or read from `config/discriminator_input.toml`, followed by the animation towards the native state. |
| `folding_path.py` | `discriminator_score` | Shortest path to the nearest native state, computed with the exact oracle. |
| `gui.py` | `gui`, `discriminator_score` | Tkinter windows: chain editor and `FoldingAnimation` animation (with GIF saving). |
| `conformation.py` | `gui`, `simulate` | Chain logic without graphics: move validation, default conformation, JSON reading/writing. |
| `simulation.py` | `simulate` | REMC on the initial conformation and saving of the minimum found. |
| `verification.py` | `verify` | Comparison of the REMC result with exact enumeration. |
| `policy_dataset.py` | `policy_dataset` | Dataset for the folding policy; utility functions (for example `contact_map`) also used by the discriminator. |
| `ml.py` | `train` | Policy model: not yet implemented. |

`torch` is imported only by the `discriminator_train` and `discriminator_score` phases, so the other phases also run without the `ml` group.

## Usage

```bash
python -m hpfold --config config/default.toml discriminator_train
```

```python
from hpfold import discriminator_model as dm

model = dm.load_model("data/output/models/discriminator")
p = dm.score(model, sequences, coordinates)  # coordinates: (S, N, 2)
```

Full list of the phases in the [main README](../README.md#running-from-the-terminal).
