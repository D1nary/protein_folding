# `tests/`

Automated tests of the project, run with `pytest` (configured in `pyproject.toml` with `testpaths = ["tests"]`).

| File | What it checks |
|---|---|
| `test_smoke.py` | The C++ extension imports and the default configuration can be read. |
| `test_core.py` | C++ core: energy, VSHD moves and pull moves on hand-built cases, validity of moves along a random walk. |
| `test_conformation.py` | Chain logic: valid and invalid moves, undo, H/P types, JSON files. |
| `test_gui.py` | Parts of `gui.py` that do not open windows: pixel/node conversion, default protein, contacts and animation frames. |
| `test_simulation.py` | REMC: temperature ladder, invalid parameters, reproducibility with the same `seed`, minimum energy, exact enumeration, `verify` phase. |
| `test_policy_dataset.py` | Action space, canonical key, symmetries, oracle distances, policy dataset files. |
| `test_discriminator_dataset.py` | Energies, kinetic traps, tiers, 1:1 balancing, duplicates, split by sequence, reproducibility. |
| `test_discriminator_model.py` | Features, absence of fields that leak the answer, invariance under symmetries, AUROC, saving and reloading the model, reproducibility. Skipped if `torch` is not installed. |
| `test_discriminator_score.py` | `discriminator_score` phase: reading and checking the input conformation, judgment outputs, command-line phase. Skipped if `torch` or `tkinter` is missing. |
| `test_folding_path.py` | Path to the native state: each move reduces the distance by one, it ends in a native state, it is deterministic. |

## Usage

From the project root, with the `dev` dependencies (and `ml` for the model tests) installed:

```bash
pytest
```

```bash
pytest tests/test_core.py -v
```
