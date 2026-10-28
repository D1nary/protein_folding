# `notes/`

Documentazione di progetto in italiano: teoria, scelte di progetto, formato dei dati e schede dei file di codice. Non viene letta dal codice.

## Da dove partire

| Nota | Argomento |
|---|---|
| [`commands.md`](commands.md) | Comandi da terminale di ogni fase: input, parametri, output. |
| [`discriminator_dataset.md`](discriminator_dataset.md) | Dataset di nativi e decoy: generazione, fasce, trappole, file, parametri, limiti. |
| [`discriminator_model.md`](discriminator_model.md) | Modello: input, architettura, training, baseline, metriche, risultati, uso. |
| [`architettura_ML.md`](architettura_ML.md) | Ragionamento alla base dell'architettura del modello. |
| [`output_parameters.md`](output_parameters.md) | Significato delle uscite di `discriminator_score`. |
| [`results_test_training.md`](results_test_training.md) | Analisi dei risultati del training (calibrazione, errori). |

## Teoria e altri temi

| Nota | Argomento |
|---|---|
| [`samplings.md`](samplings.md) | Metodi di campionamento e algoritmo REMC (Thachuk et al., 2007). |
| [`move_set.md`](move_set.md) | Pull moves e insiemi di mosse su reticolo. |
| [`policy_dataset.md`](policy_dataset.md) | Dataset per la policy di folding e oracolo esatto. |
| `for_me.md` | Promemoria personale dei comandi. |

## Sottocartelle

| Cartella | Contenuto |
|---|---|
| `code_info/` | Spiegazioni di build e strumenti (`pyproject.toml`, CMake, pytest), del formato della conformazione iniziale e dei nomi `hpfold` / `_core` / `hp_core`. |
| `file_functioning/` | Una scheda per ogni header di `cpp/include/hp/`: scopo, dipendenze, funzioni. |
| `python/hpfold/` | Schede di `conformation.py`, `policy_dataset.py` e `simulation.py`. |
