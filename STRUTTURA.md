# Struttura del progetto

Per ogni cartella: a cosa serve e cosa contengono i suoi file.

## Radice del progetto

Contiene i file che descrivono il progetto nel suo insieme e come costruirlo.

File:

- `README.md`: Descrizione del progetto, istruzioni di installazione e d'uso.
- `STRUTTURA.md`: Questo file.
- `pyproject.toml`: Definisce il pacchetto Python `hpfold`: nome, dipendenze (`numpy`, `pytest`) e sistema di build (scikit-build-core + pybind11). È il file che `pip install -e .` legge.
- `CMakeLists.txt`: Istruzioni di compilazione del C++: elenca i file `.cpp` da compilare e produce il modulo Python `hpfold._core`. Viene eseguito da `pip install`, non a mano.



## `config/`

Contiene i file di configurazione. Tutti i parametri della simulazione stanno qui: nel codice non ci sono valori hard-coded.

File:

- `default.toml`: Parametri con i valori di default: dimensioni della griglia, lunghezza e sequenza della proteina, parametri REMC, percorsi di input e output. Per un esperimento diverso si copia questo file e si passa la copia con `--config`.

## `cpp/`

Contiene tutto il codice C++: la ricerca dello stato nativo con REMC e il
salvataggio delle configurazioni intermedie. È divisa in tre sottocartelle.

### `cpp/include/hp/`

Header (`.hpp`): dichiarano cosa offre ogni componente del nucleo C++. Tutto il
codice sta nel namespace `hp`.

File:

- `conformation.hpp`: Modello HP: la catena sul reticolo (sequenza e posizioni dei residui), il controllo di validità e il calcolo dell'energia.
- `pull_moves.hpp`: Set di mosse (pull moves) con cui si modifica una conformazione.
- `remc.hpp`: Algoritmo Replica Exchange Monte Carlo: campionamento locale e scambi tra repliche.
- `sample_writer.hpp`: Salvataggio su file delle configurazioni visitate durante la simulazione.

### `cpp/src/`

Implementazioni (`.cpp`): un file per ogni header di `cpp/include/hp/`, con lo
stesso nome, che contiene il codice vero e proprio di quel componente
(`conformation.cpp`, `pull_moves.cpp`, `remc.cpp`, `sample_writer.cpp`).

### `cpp/bindings/`

Collegamento tra C++ e Python.

File:

- `bindings.cpp`: Wrapper pybind11: elenca le funzioni e le classi C++ da rendere chiamabili da Python, sotto forma di modulo `hpfold._core`.

## `python/hpfold/`

Pacchetto Python `hpfold`, che orchestra l'intera pipeline. Ogni fase del
progetto ha il suo modulo, con una funzione `run(config)` come punto d'ingresso.

File:

- `__init__.py`: Rende la cartella un pacchetto Python importabile.
- `__main__.py`: Riga di comando: `python -m hpfold --config <file> <fase>` legge la configurazione e avvia la fase richiesta.
- `config.py`: Lettura del file di configurazione TOML.
- `gui.py`: Fase 0: interfaccia per disegnare a mano la configurazione iniziale.
- `simulation.py`: Fase 1: avvia la simulazione REMC chiamando il codice C++ (`hpfold._core`).
- `dataset.py`: Fase 2: costruisce il dataset etichettato (conformazioni native e decoy).
- `ml.py`: Fase 3: addestramento del modello di machine learning.

## `data/`

Contiene i dati letti e prodotti dalla pipeline. I percorsi usati sono quelli indicati nel file di configurazione.

### `data/input/`

Dati di ingresso della simulazione: qui la GUI salva la configurazione iniziale
della proteina. La cartella è versionata con git.

### `data/output/`

Risultati delle simulazioni: configurazioni intermedie salvate dal C++ e dataset. Il contenuto è rigenerabile e quindi non è versionato con git.

I file `.gitkeep` nelle due cartelle sono vuoti: servono solo a far sì che git conservi le cartelle anche quando non contengono altro.

## `build/`

Cartella creata da `pip install -e .`: contiene i file intermedi e il risultato della compilazione del C++. Si può cancellare senza problemi (viene ricreata alla compilazione successiva) e non è versionata con git.
