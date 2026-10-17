# `python/hpfold/dataset.py`

## 1. Scopo generale del file

È il modulo della **fase 2**: costruisce il dataset per la policy di folding e lo rilegge. Come `simulation.py`, lascia al core C++ il lavoro pesante (spazio delle azioni, chiavi canoniche, grafo, oracolo, pulizia delle traiettorie). In Python restano:

- la scelta delle sequenze e degli stati;
- gli split;
- la scrittura dei file;
- le funzioni per usarli in fase 3.

Le scelte di progetto e il formato dei file sono descritti in [notes/dataset.md](../../dataset.md).

**Come si inserisce nel progetto:**

- **Usa** l'estensione `hpfold._core` (`SampleRecorder`, `run_remc`, `StateGraph`, `Oracle`, `prune_trajectory`, `successors`, `action_table`, …) e numpy.
- **Legge** dalla configurazione le sezioni `[protein]` (`length`, `sequence`), `[lattice]`, `[dataset]`, `[dataset.remc]` e `[output]` (`dataset_dir`).
- **Usato da** [\_\_main\_\_.py](../../../python/hpfold/__main__.py) (fase `dataset`) e da [test_dataset.py](../../../tests/test_dataset.py). In fase 3, `ml.py` leggerà il dataset con `load_split`.

## 2. Componenti

### Generazione

- **`candidate_sequences`**: flusso di sequenze casuali diverse. Una sequenza e la sua inversa contano come una sola.
- **`assign_splits`**: split per sequenza (train / val / test).
- **`sample_remc`**: REMC di una sequenza con un `SampleRecorder`, partendo dalla catena dritta (`straight_chain`).
- **`sequence_records`**: record, percorsi e statistiche delle traiettorie di una sequenza. È il cuore della pipeline (vedi [notes/dataset.md](../../dataset.md), §6).
- **`build_dataset`**, **`run`**: generazione di tutte le sequenze, scrittura dei file, riepilogo stampato.

### Lettura

- **`load_meta`**, **`load_sequences`**, **`load_split`**: rileggono i file. `load_split` scompatta la maschera in bool e aggiunge la sequenza `hp` di ogni record.
- **`paths`**: percorsi di uno split, come tuple (tipo, indici dei record, azioni).
- **`optimal_actions`**, **`soft_targets`**: target multi-label o distribuzione sulle mosse ottime, eventualmente pesata con il ΔE.

### Codifiche e simmetrie

- **`keys_to_turns`**, **`turns_to_coordinates`**, **`contact_map`**, **`lattice_grid`**: dalla chiave ai turni, ai turni alle coordinate, e poi alla matrice di contatto o all'immagine del reticolo.
- **`mirror`**, **`augment`**, **`rotate_coordinates`**: augmentation per simmetria al caricamento. Lo specchio scambia L/R e permuta gli slot con `mirror_permutation`. Le rotazioni servono solo alla codifica a griglia.
