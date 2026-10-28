# `config/`

File di configurazione TOML di tutte le fasi. Nel codice non ci sono parametri hard-coded: ogni fase legge il file passato con `--config`.

| File | Contenuto |
|---|---|
| `default.toml` | Parametri di default di tutte le fasi: reticolo (`[lattice]`), proteina (`[protein]`), GUI e animazione (`[gui]`), REMC della fase `simulate` (`[remc]`), verifica esatta (`[exact]`), percorsi di output (`[output]`), dataset della policy (`[policy_dataset]`, `[policy_dataset.remc]`), dataset del discriminator (`[discriminator_dataset]`, `[discriminator_dataset.remc]`) e modello (`[discriminator_model]`). Ogni parametro è commentato nel file. |
| `discriminator_input.toml` | Conformazione giudicata da `discriminator_score` quando non la si disegna nella GUI: `sequence` (stringa H/P) e `coordinates` (coppie `[x, y]` in ordine di catena). La catena deve avere `discriminator_dataset.length` residui. Il percorso è `discriminator_model.input_conformation` in `default.toml`. |

## Uso

```bash
python -m hpfold --config config/default.toml <fase>
```

Per un esperimento diverso si copia `default.toml` e si passa la copia con `--config`. I percorsi relativi sono risolti rispetto alla directory da cui si lancia il comando (la radice del progetto). La lettura è in [`python/hpfold/config.py`](../python/hpfold/config.py).
