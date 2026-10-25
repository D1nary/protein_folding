# `python/hpfold/simulation.py`

## 1. Scopo generale del file

È il modulo che realizza la **fase 1** della pipeline: la ricerca dello stato nativo con il Replica Exchange Monte Carlo. Non contiene l'algoritmo, che vive nel core C++: è un involucro Python sottile che traduce il dizionario di configurazione nei parametri della simulazione, la lancia e salva su file la conformazione a energia più bassa trovata.

**Come si inserisce nel progetto:**

- **Usa** l'estensione C++ `hpfold._core` (definita in [bindings.cpp](../../../cpp/bindings/bindings.cpp)), da cui prende `Conformation`, `RemcParams`, `RemcResult` e `run_remc`, e [conformation.py](../../../python/hpfold/conformation.py), da cui prende `load_conformation` e `save_conformation` per leggere e scrivere i file JSON.
- **Legge** dalla configurazione ([default.toml](../../../config/default.toml)) le sezioni `[lattice]`, `[remc]`, `[protein]` (solo `initial_conformation`) e `[output]` (solo `native_conformation`).
- **Usato da** [\_\_main\_\_.py](../../../python/hpfold/__main__.py), che associa la fase `simulate` della riga di comando a `simulation.run`, e dai test in [test_simulation.py](../../../tests/test_simulation.py).
- **A monte** c'è la fase 0 ([gui.py](../../../python/hpfold/gui.py)), che scrive la conformazione iniziale; **a valle** c'è la fase `verify` ([verification.py](../../../python/hpfold/verification.py)), che rilegge il file salvato qui e lo confronta con il minimo esatto.

## 2. Componenti

Il file definisce due funzioni.

- **`simulate(config)`** — Esegue il REMC a partire dalla conformazione iniziale e restituisce il `RemcResult` prodotto dal core C++, senza scrivere né stampare nulla. Nel codice è chiamata solo da `run`.

  Svolge tre passi:

  - **Lettura della conformazione iniziale** — sequenza e coordinate vengono dal file scritto nella fase 0 (`protein.initial_conformation`), non da `protein.sequence`.
  - **Costruzione di `_core.Conformation`** — con le dimensioni del reticolo prese da `[lattice]`. È il costruttore C++ a verificare che la catena sia un cammino auto-evitante dentro il reticolo: se non lo è lancia `ValueError`.
  - **Costruzione di `_core.RemcParams` e chiamata a `_core.run_remc`** — i sette parametri (`n_replicas`, `t_min`, `t_max`, `local_steps`, `n_cycles`, `rho`, `seed`) sono copiati uno a uno dalla sezione `[remc]`. Anche qui `ValueError` se i parametri non sono validi.

- **`run(config)`** — È il punto d'ingresso della fase, quello invocato dalla riga di comando. Chiama `simulate`, salva la migliore conformazione (sequenza, coordinate ed energia) nel file `output.native_conformation` e stampa un riepilogo: sequenza, energia minima, scambi accettati su scambi tentati e percorso del file salvato.

## Come avviene la conversione dei parametri dal file toml a c++

Avviene in tre passaggi, e il C++ non vede mai il file TOML:

2. **TOML → dizionario Python.** `load_config` in [config.py](https://claude.ai/epitaxy/python/hpfold/config.py) legge il file con `tomllib` e restituisce un dizionario annidato, ad esempio `config["remc"]["n_replicas"] == 5`. I tipi sono già quelli giusti: interi, float e stringhe.

3. **Dizionario → oggetto `RemcParams`.** `simulate` in [simulation.py](https://claude.ai/epitaxy/python/hpfold/simulation.py) prende i sette valori da `config["remc"]` e li passa al costruttore `_core.RemcParams(...)`.

4. **Python → C++.** Qui lavora pybind11, tramite [bindings.cpp](https://claude.ai/epitaxy/cpp/bindings/bindings.cpp): `_core.RemcParams` è la struct C++ `hp::RemcParams` esposta a Python, quindi crearla da Python crea una vera struct C++ in memoria. pybind11 converte ogni valore nel tipo C++ corrispondente (`int` Python → `int`, `float` Python → `double`) e, alla chiamata di `run_remc`, passa la struct direttamente alla funzione C++.

Lo stesso vale per la conformazione iniziale: sequenza e coordinate vengono lette dal JSON in Python e pybind11 le converte in `std::string` e `std::vector<Coord>` quando si costruisce `_core.Conformation`.

*Nota:* le chiavi `remc.save_every` e `output.samples_dir` sono presenti in [default.toml](../../../config/default.toml) ma non vengono lette da nessun file: la fase 2 ([policy_dataset.py](../../../python/hpfold/policy_dataset.py)) lancia una propria REMC per ogni sequenza, con i parametri di `[policy_dataset.remc]`, e registra in memoria gli stati visitati con un `SampleRecorder`. Dal `RemcResult` vengono inoltre usati solo `best`, `best_energy`, `swap_attempts` e `swap_accepted`; `temperatures` e `final_energies` restano inutilizzati.

## In sintesi

Il file riceve da `__main__.py` il dizionario di configurazione già caricato (non apre mai il TOML), passa i parametri tramite il binding al core C++, che li riceve come struct `RemcParams` (prende i parametri dal file di configurazione e li trasforma in un formato leggibile da c++ tramite il binding), lancia da Python la simulazione REMC in C++ e infine salva su file la conformazione a energia più bassa trovata.
