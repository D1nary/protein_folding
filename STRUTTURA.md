# Struttura del progetto

Per ogni cartella: a cosa serve e cosa contengono i suoi file.

## Radice del progetto

Contiene i file che descrivono il progetto nel suo insieme e come costruirlo.

File:

- `README.md`: Descrizione del progetto, istruzioni di installazione e d'uso.
- `STRUTTURA.md`: Questo file.
- `pyproject.toml`: Definisce il pacchetto Python `hpfold`: nome, dipendenze (`numpy`, `pytest`) e sistema di build (scikit-build-core + pybind11). È il file che `pip install -e .` legge.
- `CMakeLists.txt`: Istruzioni di compilazione del C++: elenca i file `.cpp` da compilare e produce il modulo Python `hpfold._core`. Viene eseguito da `pip install`, non a mano.
- `.clangd`: Indica a clangd (il language server C++ dell'editor) dove trovare `compile_commands.json`, cioè in `build/clangd`. Serve solo alla diagnostica nell'editor, non alla compilazione.



## `config/`

Contiene i file di configurazione. Tutti i parametri della simulazione stanno qui: nel codice non ci sono valori hard-coded.

File:

- `default.toml`: Parametri con i valori di default: dimensioni della griglia, lunghezza e sequenza della proteina di default, aspetto della GUI, parametri REMC (repliche, temperature, passi, `rho`, `seed`), lunghezza massima verificata con l'enumerazione esatta, parametri del dataset (`[dataset]` e la REMC più corta `[dataset.remc]`), percorsi di input e output. Per un esperimento diverso si copia questo file e si passa la copia con `--config`.

## `cpp/`

Contiene tutto il codice C++: la ricerca dello stato nativo con REMC (fase 1)
e le parti pesanti del dataset (fase 2): spazio delle azioni, forma canonica,
grafo delle mosse con l'oracolo e registrazione delle traiettorie REMC. È
divisa in tre sottocartelle.

### `cpp/include/hp/`

Header (`.hpp`): dichiarano cosa offre ogni componente del nucleo C++. Tutto il
codice sta nel namespace `hp`.

File:

- `conformation.hpp`: Modello HP: la classe `Conformation` (sequenza e posizioni dei residui su una griglia finita), il controllo di validità, il calcolo dell'energia e l'applicazione di una mossa. Definisce anche i tipi `Coord` (nodo) e `Move` (elenco dei residui spostati con la nuova posizione).
- `vshd_moves.hpp`: Mosse VSHD: end move, corner move e crankshaft. `vshd_moves(conformazione, k)` restituisce le mosse ammissibili sul residuo `k`.
- `pull_moves.hpp`: Pull moves di Lesh et al., sui residui interni (in entrambe le direzioni) e sui due estremi. `pull_moves(conformazione, k)` restituisce le mosse ammissibili sul residuo `k`.
- `remc.hpp`: Algoritmo Replica Exchange Monte Carlo, variante ibrida REMC_m: parametri (`RemcParams`), risultato (`RemcResult`), scala lineare delle temperature e `run_remc`, con ricerca locale Metropolis e scambi tra repliche vicine. Accetta un `SampleRecorder` opzionale.
- `exact.hpp`: Enumerazione esatta: `exact_ground_state(sequenza, larghezza, altezza)` calcola la vera energia minima di una sequenza corta visitando tutte le camminate autoevitanti (ricerca in profondità con branch and bound). Serve a verificare il minimo trovato dal REMC.
- `sample_writer.hpp`: `SampleRecorder`, che durante la REMC registra in memoria la traiettoria di ogni walker (mossa per mossa) e gli snapshot delle repliche; i file li scrive Python (fase 2).
- `canonical.hpp`: Forma canonica di una conformazione (turni FORWARD/LEFT/RIGHT, invariante per traslazioni, rotazioni e riflessioni) e la sua chiave a 64 bit; ricostruzione di una conformazione centrata sul reticolo libero.
- `actions.hpp`: Spazio delle azioni del dataset: 6N + 11 slot (residuo, tipo di mossa) che corrispondono 1:1 alle mosse di `vshd_moves` e `pull_moves`, con la permutazione degli slot sotto riflessione.
- `state_graph.hpp`: Oracolo del dataset: enumerazione di tutte le conformazioni, grafo delle mosse, distanze dallo stato nativo (BFS), etichette di ogni azione, percorsi ottimi e pulizia delle traiettorie REMC.

### `cpp/src/`

Implementazioni (`.cpp`): un file per ogni header di `cpp/include/hp/`, con lo
stesso nome, che contiene il codice vero e proprio di quel componente
(`conformation.cpp`, `vshd_moves.cpp`, `pull_moves.cpp`, `remc.cpp`,
`exact.cpp`, `sample_writer.cpp`, `canonical.cpp`, `actions.cpp`,
`state_graph.cpp`).

### `cpp/bindings/`

Collegamento tra C++ e Python.

File:

- `bindings.cpp`: Wrapper pybind11: elenca le funzioni e le classi C++ da rendere chiamabili da Python, sotto forma di modulo `hpfold._core`. Espone `Conformation`, `vshd_moves`, `pull_moves`, `RemcParams`, `RemcResult`, `temperature_ladder`, `run_remc`, `ExactResult` e `exact_ground_state`, e per la fase 2 `SampleRecorder`, le funzioni della forma canonica e dello spazio delle azioni, `StateGraph`, `Oracle` e `prune_trajectory`. I risultati grandi (chiavi, etichette) arrivano a Python come array numpy.

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
- `verification.py`: Verifica della fase 1, eseguita solo su richiesta (`verify`): legge la conformazione salvata dal REMC e ne confronta l'energia con il minimo esatto calcolato per enumerazione, se la catena non supera `exact.max_length`.
- `dataset.py`: Fase 2: costruisce il dataset per la policy di folding (stati da REMC, percorsi ottimi verso lo stato nativo, stati di recupero, etichette dell'oracolo C++), lo divide in split per sequenza e lo salva; contiene anche le funzioni per rileggerlo (`load_split`), per ricavare le codifiche (coordinate, matrice di contatto, griglia) e per l'augmentation per simmetria. Vedi `notes/dataset.md`.
- `ml.py`: Fase 3: addestramento del modello di machine learning.

## `tests/`

Test automatici, eseguiti con `pytest`.

File:

- `test_smoke.py`: Controlli minimi: l'estensione C++ si importa e la configurazione di default si legge.
- `test_conformation.py`: Logica della fase 0: mosse valide e non valide, annullamento, tipi H/P, conformazione di default, file JSON.
- `test_gui.py`: Parti di `gui.py` che non richiedono una finestra: conversione pixel/nodo e salvataggio della proteina di default.
- `test_core.py`: Nucleo C++: calcolo dell'energia, mosse VSHD e pull move su casi costruiti a mano, e validità di ogni mossa (catena connessa, autoevitante, dentro il reticolo) lungo una camminata casuale.
- `test_simulation.py`: Fase 1: scala delle temperature, parametri non validi, riproducibilità a parità di `seed`, raggiungimento dell'energia minima, file di output, enumerazione esatta e fase `verify`.
- `test_dataset.py`: Fase 2: corrispondenza 1:1 tra slot e move set, validità delle maschere, invarianza della chiave canonica e correttezza delle simmetrie, distanze dell'oracolo, pulizia delle traiettorie, e sui file generati (oracolo esatto e grafo REMC): ogni percorso termina nello stato nativo, split senza sequenze in comune, augmentation.

## `data/`

Contiene i dati letti e prodotti dalla pipeline. I percorsi usati sono quelli indicati nel file di configurazione.

### `data/input/`

Dati di ingresso della simulazione: qui la GUI salva la configurazione iniziale
della proteina (`initial_conformation.json`, formato descritto in
`notes/code_info/initial_conformation.md`). La cartella è versionata con git.

### `data/output/`

Risultati delle simulazioni: la conformazione a energia minima trovata dalla fase 1 (`native_conformation.json`, stesso formato del file di input con in più la chiave `energy`) e il dataset della fase 2 (`dataset/`: `meta.json`, `sequences.npz`, uno `.npz` per split e `trajectories.csv`, formato in `notes/dataset.md`). Il contenuto è rigenerabile e quindi non è versionato con git.

I file `.gitkeep` nelle due cartelle sono vuoti: servono solo a far sì che git conservi le cartelle anche quando non contengono altro.

## `build/`

Cartella creata da `pip install -e .`: contiene i file intermedi e il risultato della compilazione del C++. Si può cancellare senza problemi (viene ricreata alla compilazione successiva) e non è versionata con git.
