# `python/hpfold/`

Pacchetto Python `hpfold`: riga di comando, orchestrazione della pipeline, dataset, modello ML e GUI. Il lavoro pesante è delegato al core C++ tramite il modulo compilato `hpfold._core`. Ogni fase ha un modulo con una funzione `run(config)` come punto d'ingresso.

| File | Fase | Ruolo |
|---|---|---|
| `__main__.py` | — | CLI `python -m hpfold --config <file> <fase>`: associa ogni nome di fase alla sua `run`. |
| `config.py` | — | Lettura del TOML di configurazione (`load_config`). |
| `discriminator_dataset.py` | `discriminator_dataset` | Dataset di nativi e decoy: REMC per sequenza, fasce `near`/`mid`/`far`, trappole cinetiche, bilanciamento 1:1, split per sequenza, CSV e report. |
| `discriminator_model.py` | `discriminator_train` | Modello PyTorch (GNN sui contatti + CNN 1D sulla sequenza, teste ausiliarie per energia, `E_min` e gap), baseline `B0a`/`B0b`/`B1b`, training con early stopping, metriche, `load_model` e `score`. |
| `discriminator_score.py` | `discriminator_score` | Giudizio di una conformazione disegnata o letta da `config/discriminator_input.toml`, poi animazione verso lo stato nativo. |
| `folding_path.py` | `discriminator_score` | Percorso più breve verso lo stato nativo più vicino, calcolato con l'oracolo esatto. |
| `gui.py` | `gui`, `discriminator_score` | Finestre Tkinter: editor della catena e animazione `FoldingAnimation` (con salvataggio GIF). |
| `conformation.py` | `gui`, `simulate` | Logica della catena senza grafica: validazione delle mosse, conformazione di default, lettura/scrittura JSON. |
| `simulation.py` | `simulate` | REMC sulla conformazione iniziale e salvataggio del minimo trovato. |
| `verification.py` | `verify` | Confronto del risultato REMC con l'enumerazione esatta. |
| `policy_dataset.py` | `policy_dataset` | Dataset per la policy di folding; funzioni di utilità (per esempio `contact_map`) usate anche dal discriminator. |
| `ml.py` | `train` | Modello della policy: non ancora implementato. |

`torch` è importato solo dalle fasi `discriminator_train` e `discriminator_score`, così le altre partono anche senza il gruppo `ml`.

## Uso

```bash
python -m hpfold --config config/default.toml discriminator_train
```

```python
from hpfold import discriminator_model as dm

model = dm.load_model("data/output/models/discriminator")
p = dm.score(model, sequences, coordinates)  # coordinates: (S, N, 2)
```

Elenco completo delle fasi nel [README principale](../README.md#comandi).
