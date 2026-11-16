# `data/`

Data read and produced by the pipeline. The actual paths are those in `[protein]` and `[output]` in [`config/default.toml`](../config/default.toml).

## `input/` (versioned)

| File | Produced by | Read by |
|---|---|---|
| `initial_conformation.json` | `gui` phase | `simulate` phase |

It is a JSON object with two keys: `sequence`, a string of `H` and `P`, and `coordinates`, a list of `[x, y]` pairs in chain order (pair `i` is the position of residue `i`).

## `output/` (not versioned, regenerable)

| Path | Produced by | Contents |
|---|---|---|
| `datasets/discriminator_dataset/` | `discriminator_dataset` | `train.csv`, `val.csv`, `test.csv` (one row per example: `seq_id`, `sequence`, `split`, `label`, `energy`, `e_min`, `gap`, `tier`, `kinetic_trap`, `native_overlap`, `turns`, `coordinates`, `key`), `sequences.csv`, `provenance.csv`, `meta.json`, `report.txt` |
| `models/discriminator/` | `discriminator_train` | `discriminator.pt` (weights and hyperparameters), `metrics.json`, `report.txt`, `history.csv`, `test_predictions.csv` |
| `animations/` | `discriminator_score` (**Save GIF…** button) | GIF `folding_<sequence>.gif` |
| `native_conformation.json` | `simulate` | minimum-energy conformation, with the `energy` key |
| `datasets/policy_dataset/` | `policy_dataset` | `meta.json`, `sequences.npz`, one `.npz` per split, `trajectories.csv` |

Fields such as `energy`, `e_min`, `gap` and `tier` contain the answer and must **not** be used as features: the model's input is only the sequence and the conformation.

The `.gitkeep` files are empty: they make git keep the folders.
