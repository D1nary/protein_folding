# `pyproject.toml` (root)

[pyproject.toml](https://claude.ai/epitaxy/pyproject.toml) è il file di configurazione standard di un progetto Python: dice a `pip` **come costruire** il pacchetto, **cosa contiene** e **come configurare gli strumenti** che ci lavorano sopra. Nel tuo caso è importante perché `hpfold` non è puro Python: ha una parte C++ che va compilata.

Quando lanci `pip install .`, pip legge il file e procede sezione per sezione.

## `[build-system]`: come si costruisce il pacchetto

```toml
requires = ["scikit-build-core>=0.10", "pybind11>=3.0"]
build-backend = "scikit_build_core.build"
```

- **`requires`**: i pacchetti necessari *solo per la build*. pip li installa in un ambiente temporaneo isolato. Sono gli strumenti che servono solo per *compilare e impacchettare* il progetto (qui `scikit-build-core` e `pybind11`): pip li installa in un ambiente temporaneo, li usa per produrre il pacchetto e poi li butta, quindi non restano nel tuo ambiente e non servono per eseguire `hpfold`, a differenza di `numpy`.
- **`build-backend`**: a chi pip delega la costruzione. Qui è `scikit-build-core`, che fa da ponte tra pip e CMake: lancia il tuo [CMakeLists.txt](https://claude.ai/epitaxy/CMakeLists.txt), compila il C++ in `cpp/` e, tramite `pybind11`, produce un modulo importabile da Python.

## `[project]`: i metadati del pacchetto

- **`name`, `version`, `description`, `readme`**: identità del pacchetto; `name` è il nome con cui lo installi.
- **`requires-python = ">=3.11"`**: pip rifiuta l'installazione su versioni più vecchie.
- **`dependencies = ["numpy"]`**: dipendenze *a runtime*, installate automaticamente insieme al pacchetto.

## `[project.optional-dependencies]`: gli extra

```toml
dev = ["pytest"]
```

Sono gruppi di dipendenze che si installano solo su richiesta, con `pip install ".[dev]"`. Chi usa solo il pacchetto non si porta dietro `pytest`. 

Il TODO prevede un futuro gruppo `ml` con la stessa logica.

## `[tool.*]`: configurazione degli strumenti

Ogni strumento legge solo la propria sottosezione.

**`[tool.scikit-build]`**

- **`minimum-version = "build-system.requires"`**: ricava la versione minima di scikit-build-core dal vincolo scritto sopra (`0.10`), così non la ripeti in due posti.
- **`cmake.build-type = "Release"`**: compila con le ottimizzazioni, che contano per un campionamento Monte Carlo. Il compilatore C++ riorganizza il codice per renderlo molto più veloce (a scapito della facilità di debug), e questo conta perché un campionamento Monte Carlo ripete le stesse operazioni milioni di volte: anche un piccolo guadagno per singola mossa diventa una grande differenza sul tempo totale.
- **`build-dir = "build/{wheel_tag}"`**: tiene i file di compilazione in una cartella persistente, separata per versione di Python e piattaforma, così le ricompilazioni successive sono incrementali. La classificazione dei diversi file di compilaizone avviene tramite il segnaposto `{wheel_tag}`. L’etichetta dipende solo dall’ambiente (versione di Python, ABI, piattaforma), non dalla singola compilazione, quindi finché compili con lo stesso Python sulla stessa macchina la cartella è sempre la stessa e viene riutilizzata. Una nuova sottocartella nasce solo se cambi ambiente, ad esempio passando da Python 3.11 a 3.12.
- **`wheel.packages = ["python/hpfold"]`**: indica dove sta il codice Python da includere, perché non è nella posizione predefinita.

**`[tool.pytest.ini_options]`**

- **`testpaths = ["tests"]`**: lanciando `pytest` senza argomenti, cerca i test solo in `tests/`.

## In pratica

Per installare il progetto con gli strumenti di sviluppo:

```bash
pip install -e ".[dev]"
```

Con `-e` (modalità editable) le modifiche ai file Python sono subito attive, mentre dopo ogni modifica al C++ devi rilanciare il comando per ricompilare.

# `CMakeLists.txt`

CMake non compila il codice: è un **generatore di sistemi di build**. Nel `CMakeLists.txt` descrivi *cosa* vuoi costruire (un eseguibile o una libreria), e CMake produce i file per lo strumento di build (Make, Ninja, ecc.) che invoca il compilatore, adattandoli al compilatore e al sistema operativo in uso.

Il flusso ha due fasi:

2. **Configurazione**: CMake legge il `CMakeLists.txt`, trova compilatore e dipendenze, e genera i file di build in una cartella separata (`build/`).
3. **Build**: lo strumento generato compila e collega i sorgenti.

Il concetto centrale è il **target**:  Essi sono prodotto della futura compilazione, cioè un eseguibile o una libreria. A ciascun target si associano le sue impostazioni: dove cercare gli header, quali opzioni passare al compilatore, quali librerie collegare. Con `PRIVATE` un’impostazione vale solo per quel target; con `PUBLIC` vale anche per i target che lo collegano.



## Cosa fa nel tuo progetto (file nella root)

Il tuo [CMakeLists.txt](https://claude.ai/epitaxy/CMakeLists.txt) costruisce il codice C++ della simulazione e lo rende importabile da Python come `hpfold._core`. Definisce due target, uno dentro l'altro:

- `_core`: modulo Python (`.so`) costruito da `bindings.cpp` con pybind11. Contiene `hp_core` al suo interno e ne espone classi e funzioni a Python: è il prodotto finale, installato come `hpfold._core`.

- `hp_core`. Lui sta dentro a `_core`. 
  **`hp_core`**: libreria statica con la logica della simulazione in C++ puro (`conformation`, `pull_moves`, `remc`, `sample_writer`). Non sa nulla di Python e non viene installata: è un prodotto intermedio (`libhp_core.a`).



Un file `.so` (*shared object*) è una **libreria condivisa** su Linux: codice già compilato in linguaggio macchina, che non è un programma avviabile da solo ma viene caricato da un altro programma mentre è in esecuzione.



Nel tuo progetto il `.so` è il modulo `_core` (vedi di seguito). Python non sa leggere il C++, ma sa caricare librerie condivise: quando esegui `import hpfold._core`, l’interprete cerca il `.so`, lo carica in memoria e chiama una funzione di inizializzazione al suo interno. Quella funzione è generata da pybind11 a partire da `bindings.cpp`, e registra le classi e le funzioni C++ come oggetti Python.




In altre parole, Python utilizza la simulazione c++ attrverso il flie `.so`



**Impostazioni generali** ([righe 1-6](https://claude.ai/epitaxy/local_a2f51f80-84ca-4a47-a72e-337d9e6aa3a0))

- Il progetto si chiama `hpfold` e usa solo C++.
- Impone C++17 standard, senza estensioni specifiche del compilatore.



**Target 1: `hp_core`, la libreria della simulazione**

- È una libreria statica con i quattro sorgenti in `cpp/src/`: `conformation`, `pull_moves`, `remc`, `sample_writer`.
- Non sa nulla di Python: è la logica pura.
- `cpp/include` è `PUBLIC`, quindi chi collega `hp_core` vede automaticamente i suoi header.
- `-Wall -Wextra` attiva i warning, solo per questa libreria.
- `POSITION_INDEPENDENT_CODE ON` (cioè `-fPIC`) serve perché la libreria finisce dentro un modulo condiviso `.so`; senza, il link fallirebbe.

**Target 2: `_core`, il modulo Python**

- `find_package(pybind11 ...)` trova pybind11, la libreria che fa da ponte C++/Python.
- ``pybind11_add_module` definisce il target `_core`: dichiara che cpp/bindings/bindings.cpp va compilato in un file .so che Python può importare.
- `target_link_libraries(_core PRIVATE hp_core)` ci mette dentro la libreria della simulazione.

`install` dichiara la regola di installazione: il `.so` (il file di binding che python può leggere) va copiato nella cartella `hpfold` (il pacchetto Python installato, quello che si usa con `import hpfold`), insieme al codice Python proveniente da `python/hpfold/` (la cartella del repository con i sorgenti Python del pacchetto), così da essere importabile come `hpfold._core` (il modulo compilato che espone a Python il codice C++ della simulazione).`.

## Come si collega al resto

Di norma non lanci CMake a mano: lo fa `pip` per te. Il [pyproject.toml](https://claude.ai/epitaxy/pyproject.toml) dichiara `scikit-build-core` come backend di build, che fa da collante:

```bash
pip install -e .
```

Questo comando esegue CMake sul `CMakeLists.txt`, compila in `build/`, e installa il pacchetto Python `hpfold` con dentro il modulo compilato `_core`.

La separazione in due target è utile: `hp_core` resta C++ puro, quindi in futuro potresti collegarlo a un eseguibile o a test C++ senza passare da Python, mentre `bindings.cpp` resta un wrapper sottile.
