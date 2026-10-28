# `data/`

Dati letti e prodotti dalla pipeline. I percorsi effettivi sono quelli di `[protein]` e `[output]` in [`config/default.toml`](../config/default.toml).

## `input/` (versionata)

| File | Prodotto da | Letto da |
|---|---|---|
| `initial_conformation.json` | fase `gui` | fase `simulate` |

È un oggetto JSON con due chiavi: `sequence`, stringa di `H` e `P`, e `coordinates`, lista di coppie `[x, y]` in ordine di catena (la coppia `i` è la posizione del residuo `i`).

## `output/` (non versionata, rigenerabile)

| Percorso | Prodotto da | Contenuto |
|---|---|---|
| `datasets/discriminator_dataset/` | `discriminator_dataset` | `train.csv`, `val.csv`, `test.csv` (una riga per esempio: `seq_id`, `sequence`, `split`, `label`, `energy`, `e_min`, `gap`, `tier`, `kinetic_trap`, `native_overlap`, `turns`, `coordinates`, `key`), `sequences.csv`, `provenance.csv`, `meta.json`, `report.txt` |
| `models/discriminator/` | `discriminator_train` | `discriminator.pt` (pesi e iperparametri), `metrics.json`, `report.txt`, `history.csv`, `test_predictions.csv` |
| `animations/` | `discriminator_score` (pulsante **Save GIF…**) | GIF `folding_<sequenza>.gif` |
| `native_conformation.json` | `simulate` | conformazione a energia minima, con la chiave `energy` |
| `datasets/policy_dataset/` | `policy_dataset` | `meta.json`, `sequences.npz`, un `.npz` per split, `trajectories.csv` |

Campi come `energy`, `e_min`, `gap` e `tier` contengono la risposta e **non** vanno usati come feature: l'input del modello è solo sequenza e conformazione.

I file `.gitkeep` sono vuoti: servono a far conservare le cartelle a git.
