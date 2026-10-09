# protein_folding

Campionamento dello spazio delle conformazioni di una proteina nel modello HP
(Hydrophobic-Polar) su reticolo 2D, con l'obiettivo di costruire un dataset
etichettato di conformazioni native e decoy su cui addestrare un modello di
machine learning.

Stato attuale: **solo struttura del progetto**. La logica delle fasi non è
ancora implementata (i file contengono scheletri e `TODO`).

## Fasi

| Fase | Contenuto | Dove |
|---|---|---|
| 0 | GUI per disegnare la configurazione iniziale | `python/hpfold/gui.py` |
| 1 | Ricerca dello stato nativo con REMC | `cpp/`, `python/hpfold/simulation.py` |
| 2 | Generazione del dataset (nativi + decoy) | `cpp/` (salvataggio), `python/hpfold/dataset.py` |
| 3 | Addestramento del modello di ML | `python/hpfold/ml.py` |

## Struttura

```
protein_folding/
├── config/
│   └── default.toml          # tutti i parametri, con i valori di default
├── cpp/                      # nucleo C++ (namespace hp)
│   ├── include/hp/
│   │   ├── conformation.hpp  # modello HP: catena sul reticolo, energia
│   │   ├── pull_moves.hpp    # set di mosse (pull moves)
│   │   ├── remc.hpp          # Replica Exchange Monte Carlo
│   │   └── sample_writer.hpp # salvataggio delle configurazioni intermedie
│   ├── src/                  # implementazioni (.cpp) degli header
│   └── bindings/
│       └── bindings.cpp      # wrapper pybind11 -> modulo hpfold._core
├── python/hpfold/            # pacchetto Python che orchestra la pipeline
│   ├── __main__.py           # python -m hpfold --config <file> <fase>
│   ├── config.py             # lettura del file di configurazione
│   ├── gui.py                # fase 0
│   ├── simulation.py         # fase 1 (chiama hpfold._core)
│   ├── dataset.py            # fase 2
│   └── ml.py                 # fase 3
├── tests/
│   └── test_smoke.py         # l'estensione si importa, la config si legge
├── data/
│   ├── input/                # configurazione iniziale disegnata con la GUI
│   └── output/               # output delle simulazioni (non versionato)
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

## Uso

```bash
python -m hpfold --config config/default.toml gui       # fase 0
python -m hpfold --config config/default.toml simulate  # fase 1
python -m hpfold --config config/default.toml dataset   # fase 2
python -m hpfold --config config/default.toml train     # fase 3
```

I percorsi relativi nel file di configurazione sono risolti rispetto alla
directory da cui si lancia il comando (la radice del progetto).
