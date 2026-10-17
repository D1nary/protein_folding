# protein_folding

Campionamento dello spazio delle conformazioni di una proteina nel modello HP
(Hydrophobic-Polar) su reticolo 2D, con l'obiettivo di costruire un dataset
etichettato di conformazioni native e decoy su cui addestrare un modello di
machine learning.

Stato attuale: **fasi 0, 1 e 2 implementate**. La fase 3 non è ancora
implementata (il file contiene uno scheletro e un `TODO`).

## Fasi

| Fase | Contenuto | Dove |
|---|---|---|
| 0 | GUI per disegnare la configurazione iniziale | `python/hpfold/gui.py`, `python/hpfold/conformation.py` |
| 1 | Ricerca dello stato nativo con REMC | `cpp/`, `python/hpfold/simulation.py` |
| 2 | Dataset per la policy di folding (stati, mosse valide, mosse ottime) | `cpp/` (azioni, oracolo, registrazione REMC), `python/hpfold/dataset.py` |
| 3 | Addestramento del modello di ML | `python/hpfold/ml.py` |

## Struttura

```
protein_folding/
├── config/
│   └── default.toml          # tutti i parametri, con i valori di default
├── cpp/                      # nucleo C++ (namespace hp)
│   ├── include/hp/
│   │   ├── conformation.hpp  # modello HP: catena sul reticolo, energia
│   │   ├── vshd_moves.hpp    # mosse VSHD (end, corner, crankshaft)
│   │   ├── pull_moves.hpp    # pull moves
│   │   ├── remc.hpp          # Replica Exchange Monte Carlo (REMC_m)
│   │   ├── exact.hpp         # enumerazione esatta dello stato nativo
│   │   ├── sample_writer.hpp # registrazione delle traiettorie REMC
│   │   ├── canonical.hpp     # forma canonica e chiave di una conformazione
│   │   ├── actions.hpp       # spazio delle azioni (6N + 11 slot)
│   │   └── state_graph.hpp   # grafo delle mosse e oracolo del dataset
│   ├── src/                  # implementazioni (.cpp) degli header
│   └── bindings/
│       └── bindings.cpp      # wrapper pybind11 -> modulo hpfold._core
├── python/hpfold/            # pacchetto Python che orchestra la pipeline
│   ├── __main__.py           # python -m hpfold --config <file> <fase>
│   ├── config.py             # lettura del file di configurazione
│   ├── conformation.py       # fase 0: logica (catena, validazione, JSON)
│   ├── gui.py                # fase 0: finestra Tkinter
│   ├── simulation.py         # fase 1 (chiama hpfold._core)
│   ├── dataset.py            # fase 2
│   └── ml.py                 # fase 3
├── tests/
│   ├── test_smoke.py         # l'estensione si importa, la config si legge
│   ├── test_conformation.py  # logica della fase 0
│   ├── test_gui.py           # parti della GUI che non aprono finestre
│   ├── test_core.py          # nucleo C++: energia, mosse VSHD e pull
│   ├── test_simulation.py    # fase 1: REMC, riproducibilità, output
│   └── test_dataset.py       # fase 2: azioni, simmetrie, oracolo, file
├── data/
│   ├── input/                # configurazione iniziale disegnata con la GUI
│   └── output/               # conformazione nativa, campioni (non versionato)
├── notes/                    # appunti teorici
├── CMakeLists.txt            # build del C++ e del modulo Python
└── pyproject.toml            # dipendenze Python e backend di build
```

## Scelte tecniche

- **Configurazione: TOML.** Si legge con `tomllib` della libreria standard, ha
  tipi non ambigui e ammette commenti. Il file è letto solo da Python, che
  passa i valori al C++ attraverso il wrapper: il C++ non ha bisogno di una
  libreria di parsing e la configurazione ha un'unica fonte.
- **Wrapper C++/Python: pybind11.** Header-only, converte automaticamente i
  contenitori STL in tipi Python e si integra con CMake.
- **Build: CMake + scikit-build-core.** `pip install` compila l'estensione e
  installa il pacchetto in un solo passo.
- **GUI: Tkinter.** Per disegnare una catena su una griglia basta un `Canvas`,
  ed è già nella libreria standard.

## Installazione (Ubuntu)

Pacchetti di sistema:

```bash
sudo apt install build-essential cmake python3-dev python3-venv python3-tk
```

Ambiente virtuale, compilazione e installazione:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pip install -e .` compila il C++ e installa il pacchetto in modalità
modificabile: le modifiche ai file Python sono subito attive, mentre dopo una
modifica al C++ va rieseguito il comando.

Verifica:

```bash
pytest
```

Diagnostica C++ nell'editor (clangd, usato ad esempio da Zed): clangd ha
bisogno di `compile_commands.json` per conoscere le directory di include. Il
file `.clangd` lo cerca in `build/clangd`, dove va generato una volta (e di
nuovo se si cancella `build/` o si aggiunge un file `.cpp`):

```bash
cmake -S . -B build/clangd -DPython_EXECUTABLE="$PWD/.venv/bin/python" \
      -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
```

## Uso

```bash
python -m hpfold --config config/default.toml gui       # fase 0
python -m hpfold --config config/default.toml simulate  # fase 1
python -m hpfold --config config/default.toml verify    # verifica della fase 1 (facoltativa)
python -m hpfold --config config/default.toml dataset   # fase 2
python -m hpfold --config config/default.toml train     # fase 3
```

I percorsi relativi nel file di configurazione sono risolti rispetto alla
directory da cui si lancia il comando (la radice del progetto).

### Fase 0: conformazione iniziale

All'avvio la GUI chiede se si vuole disegnare una nuova proteina:

- **No**: viene salvata la proteina di default, cioè `protein.sequence` con una
  forma generata automaticamente (catena a serpentina dall'angolo in alto a
  sinistra).
- **Yes**: si apre la griglia `lattice.width` × `lattice.height` e si procede in
  due step.
  1. **Disegno della catena.** Ogni click su un nodo aggiunge un residuo,
     collegato al precedente. Il nodo deve essere libero e adiacente
     (su/giù/sinistra/destra) all'ultimo residuo; il primo può andare ovunque.
     "Undo last residue" e "Clear all" correggono il disegno. La
     catena deve avere esattamente `protein.length` residui.
  2. **Assegnazione H/P.** Si sceglie il tipo attivo (H o P) e si clicca sui
     residui; un altro click cambia il tipo. Si può tornare allo step 1 per
     correggere la forma.

  "Save" si abilita quando tutti i residui hanno un tipo; dopo il salvataggio
  la finestra si chiude da sola.

In entrambi i casi il risultato è scritto in `protein.initial_conformation`
(formato in `notes/code_info/initial_conformation.md`): è da questo file, e non
da `protein.sequence`, che la fase 1 legge sequenza e coordinate. Colori e
dimensioni iniziali della finestra si regolano nella sezione `[gui]` del TOML;
ridimensionando la finestra, la griglia e i residui si adattano allo spazio
disponibile.

### Fase 1: ricerca della conformazione nativa (REMC)

```bash
python -m hpfold --config config/default.toml simulate
```

Parte dalla conformazione in `protein.initial_conformation` ed esegue il
Replica Exchange Monte Carlo di Thachuk, Shmygelska e Hoos (2007) nella
variante ibrida REMC_m, sul reticolo 2D `lattice.width` × `lattice.height`.
Al termine stampa l'energia minima e salva la conformazione corrispondente in
`output.native_conformation`, nello stesso formato del file di input con in
più la chiave `energy`.

Parametri, tutti nella sezione `[remc]` del TOML:

| Parametro | Significato |
|---|---|
| `n_replicas` | numero di repliche (almeno 2) |
| `t_min`, `t_max` | temperature della replica più fredda e più calda (`k_B = 1`) |
| `local_steps` | passi Monte Carlo locali tra due tentativi di scambio |
| `n_cycles` | numero di cicli (ricerca locale + scambi) |
| `rho` | probabilità che un passo locale usi una pull move (altrimenti VSHD) |
| `seed` | seme del generatore casuale |

Come funziona:

- **Energia**: −1 per ogni coppia di residui H adiacenti sul reticolo ma non
  consecutivi nella catena.
- **Temperature**: distribuite linearmente tra `t_min` e `t_max`. I valori di
  default sono quelli del paper divisi per 1000.
- **Passo locale**: si sceglie l'insieme di mosse (pull con probabilità `rho`,
  altrimenti VSHD), poi un residuo a caso, poi una a caso tra le mosse
  ammissibili su quel residuo; la mossa è accettata con il criterio di
  Metropolis. Se sul residuo non c'è nessuna mossa ammissibile il passo lascia
  la conformazione invariata.
- **Reticolo finito**: una mossa che porterebbe un residuo fuori dalla griglia
  non è ammissibile. La griglia va quindi tenuta ampia rispetto alla proteina.
- **Scambi**: a ogni ciclo si propongono scambi tra repliche a temperature
  vicine, alternando le coppie (1,2), (3,4), … e (2,3), (4,5), …
- **Miglior conformazione**: aggiornata a ogni mossa accettata, non solo alla
  fine della ricerca locale.
- **Arresto**: dopo `n_cycles` cicli. Tutte le repliche partono dalla stessa
  conformazione iniziale.

A parità di `seed` (e di macchina e compilatore) la simulazione è
riproducibile. `remc.save_every` e `output.samples_dir` non sono usati: la
fase 2 lancia una propria REMC per ogni sequenza (`[dataset.remc]`) e
registra gli stati visitati in memoria.

### Fase 2: dataset

```bash
python -m hpfold --config config/default.toml dataset
```

Per molte sequenze casuali di lunghezza `protein.length`, raccoglie stati
dalla REMC e li etichetta con un oracolo (ricerca in ampiezza sul grafo delle
mosse): per ogni stato le mosse valide, la distanza dallo stato nativo dopo
ogni mossa e il ΔE di ogni mossa. Fino a `dataset.exact_max_length` residui
(16) l'oracolo è esatto. Formato, scelte e lettura dei file sono in
`notes/dataset.md`.

### Verifica della fase 1: enumerazione esatta

```bash
python -m hpfold --config config/default.toml verify
```

Il REMC è un metodo stocastico: restituisce l'energia più bassa tra quelle
visitate, senza garanzia che sia il minimo. Questo comando, da lanciare dopo
`simulate`, legge la conformazione in `output.native_conformation` e ne
confronta l'energia con il minimo esatto della sequenza, calcolato enumerando
tutte le conformazioni che entrano nel reticolo (ricerca in profondità con
branch and bound). Non fa parte della simulazione: viene eseguito solo quando
lo si richiede.

Il costo cresce esponenzialmente con la lunghezza della catena, quindi le
sequenze più lunghe di `exact.max_length` non vengono verificate.
