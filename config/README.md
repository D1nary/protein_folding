# `config/`

TOML configuration files for all phases. There are no hard-coded parameters in the code: each phase reads the file passed with `--config`.

| File | Contents |
|---|---|
| `default.toml` | Default parameters for all phases: lattice (`[lattice]`), protein (`[protein]`), GUI and animation (`[gui]`), REMC of the `simulate` phase (`[remc]`), exact verification (`[exact]`), output paths (`[output]`), policy dataset (`[policy_dataset]`, `[policy_dataset.remc]`), discriminator dataset (`[discriminator_dataset]`, `[discriminator_dataset.remc]`) and model (`[discriminator_model]`). Every parameter is commented in the file. |
| `discriminator_input.toml` | Conformation judged by `discriminator_score` when it is not drawn in the GUI: `sequence` (H/P string) and `coordinates` (`[x, y]` pairs in chain order). The chain must have `discriminator_dataset.length` residues. Its path is `discriminator_model.input_conformation` in `default.toml`. |

## Usage

```bash
python -m hpfold --config config/default.toml <phase>
```

For a different experiment, copy `default.toml` and pass the copy with `--config`. Relative paths are resolved against the directory the command is run from (the project root). The reading is in [`python/hpfold/config.py`](../python/hpfold/config.py).
