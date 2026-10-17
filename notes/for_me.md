# Comandi del progetto

Da lanciare dalla radice del progetto, con il venv attivo. Dopo ogni modifica
al C++ va ricompilata l'estensione `hpfold._core`.

## Fase 0: GUI

```bash
python -m hpfold --config config/default.toml gui
```

Apre una finestra Tkinter in cui si disegna la conformazione iniziale sul
reticolo. Il risultato (sequenza + coordinate) viene salvato nel file indicato
da `protein.initial_conformation` (`data/input/initial_conformation.json`).

## Fase 1: simulazione (REMC)

```bash
python -m hpfold --config config/default.toml simulate
```

Esegue solo il REMC (variante REMC_m), partendo dalla conformazione in
`protein.initial_conformation`, non da `protein.sequence`. Usa il reticolo di
`[lattice]` e i parametri di `[remc]`. Stampa l'energia minima e salva la
conformazione migliore in `output.native_conformation`
(`data/output/native_conformation.json`).

## Fase 2: dataset

```bash
python -m hpfold --config config/default.toml dataset
```

Costruisce il dataset per la policy di folding in `output.dataset_dir`. Non usa
né la GUI né `initial_conformation.json`.

- Sequenze: `dataset.n_sequences` (100) di lunghezza `protein.length`. La prima
  è `protein.sequence` (se `include_config_sequence = true`), le altre sono
  casuali (H con probabilità `h_fraction`). Una sequenza e la sua inversa
  contano come una sola; quelle con energia minima 0 vengono scartate.
- Per ogni sequenza lancia un REMC più corto (`[dataset.remc]`) da una catena
  dritta al centro del reticolo; gli stati visitati forniscono decoy e
  traiettorie.
- Le etichette (mosse valide, distanza dal nativo, ΔE) vengono da un oracolo
  (BFS sul grafo delle mosse): esatto fino a `exact_max_length` residui,
  altrimenti calcolato sul grafo degli stati visitati dal REMC.
- Split per sequenza in train/val/test secondo `dataset.split` (80/10/10).
- Output: `train.npz`, `val.npz`, `test.npz`, `sequences.npz`,
  `trajectories.csv`, `meta.json`. Dettagli in `notes/dataset.md`.

L’algoritmo crea n_replicas copie della configurazione iniziale e assegna a ciascuna una temperatura diversa, equispaziata tra t_min e t_max. Ogni replica esegue local_steps passi di Monte Carlo Metropolis alla propria temperatura, usando pull moves con probabilità rho oppure mosse VSHD. Al termine dei passi locali si tentano scambi di conformazione tra repliche a temperature adiacenti, alternando ad ogni ciclo le coppie pari (0-1, 2-3, …) e dispari (1-2, 3-4, …) (per esempio al cycle 1 di n_cycles si tenta lo scambio solo tra coppie pari mentre al cycle 1  si tenta lo scambio solo tra coppie dispari). Ogni tentativo di scambio tra due repliche a temperature adiacenti viene accettato o rifiutato con una certa probabilità. Il ciclo si ripete n_cycles volte e si conserva la conformazione di energia minima trovata.

Alla fine dell'algoritmo REMC cosa ottengo?
Alla fine hai n_replicas temperature fisse, ognuna con la conformazione e l’energia che ci si trovano in quel momento. Le conformazioni si sono mosse lungo la scala grazie agli swap. L’output che conta per il folding è la migliore conformazione trovata durante tutta la simulazione.

Caratteristiche che deve opssedere il dataset:

- Range di energie ampio. Se il dataset contiene solo enerige vicine al folding, il modello ML non sarà in grado di individuare





Come vengono scelte le configurazioni casuali?
**1. Generazione casuale.** Ogni sequenza ha lunghezza fissa `protein.length`. Per ogni residuo si estrae un numero uniforme in [0,1): se è minore di `h_fraction` il residuo è H, altrimenti P
