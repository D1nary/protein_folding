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

- `default.toml`: Parametri con i valori di default: dimensioni della griglia, lunghezza e sequenza della proteina di default, aspetto della GUI, parametri REMC (repliche, temperature, passi, `rho`, `seed`), percorsi di input e output. Per un esperimento diverso si copia questo file e si passa la copia con `--config`.

## `cpp/`

Contiene tutto il codice C++: la ricerca dello stato nativo con REMC (fase 1,
implementata) e il salvataggio delle configurazioni intermedie (ancora da
scrivere). È divisa in tre sottocartelle.

### `cpp/include/hp/`

Header (`.hpp`): dichiarano cosa offre ogni componente del nucleo C++. Tutto il
codice sta nel namespace `hp`.

File:

- `conformation.hpp`: Modello HP: la classe `Conformation` (sequenza e posizioni dei residui su una griglia finita), il controllo di validità, il calcolo dell'energia e l'applicazione di una mossa. Definisce anche i tipi `Coord` (nodo) e `Move` (elenco dei residui spostati con la nuova posizione).
- `vshd_moves.hpp`: Mosse VSHD: end move, corner move e crankshaft. `vshd_moves(conformazione, k)` restituisce le mosse ammissibili sul residuo `k`.
- `pull_moves.hpp`: Pull moves di Lesh et al., sui residui interni (in entrambe le direzioni) e sui due estremi. `pull_moves(conformazione, k)` restituisce le mosse ammissibili sul residuo `k`.
- `remc.hpp`: Algoritmo Replica Exchange Monte Carlo, variante ibrida REMC_m: parametri (`RemcParams`), risultato (`RemcResult`), scala lineare delle temperature e `run_remc`, con ricerca locale Metropolis e scambi tra repliche vicine.
- `sample_writer.hpp`: Salvataggio su file delle configurazioni visitate durante la simulazione (non ancora implementato).

### `cpp/src/`

Implementazioni (`.cpp`): un file per ogni header di `cpp/include/hp/`, con lo
stesso nome, che contiene il codice vero e proprio di quel componente
(`conformation.cpp`, `vshd_moves.cpp`, `pull_moves.cpp`, `remc.cpp`,
`sample_writer.cpp`).

### `cpp/bindings/`

Collegamento tra C++ e Python.

File:

- `bindings.cpp`: Wrapper pybind11: elenca le funzioni e le classi C++ da rendere chiamabili da Python, sotto forma di modulo `hpfold._core`. Espone `Conformation`, `vshd_moves`, `pull_moves`, `RemcParams`, `RemcResult`, `temperature_ladder` e `run_remc`.

## `python/hpfold/`

Pacchetto Python `hpfold`, che orchestra l'intera pipeline. Ogni fase del
progetto ha il suo modulo, con una funzione `run(config)` come punto d'ingresso.

File:

- `__init__.py`: Rende la cartella un pacchetto Python importabile.
- `__main__.py`: Riga di comando: `python -m hpfold --config <file> <fase>` legge la configurazione e avvia la fase richiesta.
- `config.py`: Lettura del file di configurazione TOML.
- `conformation.py`: Logica della fase 0, senza codice grafico: stato della catena in costruzione, validazione delle mosse, conformazione di default e lettura/scrittura del file JSON (usata anche dalla fase 1 per leggere l'input e scrivere il risultato).
- `gui.py`: Fase 0: finestra Tkinter per disegnare a mano la configurazione iniziale. Contiene solo l'interfaccia e usa `conformation.py` per tutto il resto.
- `simulation.py`: Fase 1: legge la conformazione iniziale, costruisce i parametri dal file di configurazione, avvia la simulazione REMC chiamando il codice C++ (`hpfold._core`) e salva la conformazione a energia minima.
- `dataset.py`: Fase 2: costruisce il dataset etichettato (conformazioni native e decoy).
- `ml.py`: Fase 3: addestramento del modello di machine learning.

## `tests/`

Test automatici, eseguiti con `pytest`.

File:

- `test_smoke.py`: Controlli minimi: l'estensione C++ si importa e la configurazione di default si legge.
- `test_conformation.py`: Logica della fase 0: mosse valide e non valide, annullamento, tipi H/P, conformazione di default, file JSON.
- `test_gui.py`: Parti di `gui.py` che non richiedono una finestra: conversione pixel/nodo e salvataggio della proteina di default.
- `test_core.py`: Nucleo C++: calcolo dell'energia, mosse VSHD e pull move su casi costruiti a mano, e validità di ogni mossa (catena connessa, autoevitante, dentro il reticolo) lungo una camminata casuale.
- `test_simulation.py`: Fase 1: scala delle temperature, parametri non validi, riproducibilità a parità di `seed`, raggiungimento dell'energia minima e file di output.

## `data/`

Contiene i dati letti e prodotti dalla pipeline. I percorsi usati sono quelli indicati nel file di configurazione.

### `data/input/`

Dati di ingresso della simulazione: qui la GUI salva la configurazione iniziale
della proteina (`initial_conformation.json`, formato descritto in
`notes/code_info/initial_conformation.md`). La cartella è versionata con git.

### `data/output/`

Risultati delle simulazioni: la conformazione a energia minima trovata dalla fase 1 (`native_conformation.json`, stesso formato del file di input con in più la chiave `energy`), le configurazioni intermedie salvate dal C++ e il dataset. Il contenuto è rigenerabile e quindi non è versionato con git.

I file `.gitkeep` nelle due cartelle sono vuoti: servono solo a far sì che git conservi le cartelle anche quando non contengono altro.

## `build/`

Cartella creata da `pip install -e .`: contiene i file intermedi e il risultato della compilazione del C++. Si può cancellare senza problemi (viene ricreata alla compilazione successiva) e non è versionata con git.
