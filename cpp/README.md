# `cpp/`

Core C++17 del progetto: modello HP su reticolo 2D, mosse, REMC, enumerazione esatta e oracolo del grafo delle conformazioni. Non dipende da Python e non legge né scrive file: riceve i dati da Python e restituisce risultati in memoria. Tutto il codice sta nel namespace `hp`.

| Cartella | Contenuto |
|---|---|
| `include/hp/` | Header: dichiarano l'interfaccia di ogni componente. |
| `src/` | Implementazioni, un `.cpp` per ogni header con lo stesso nome. |
| `bindings/` | `bindings.cpp`: wrapper pybind11 che crea il modulo Python `hpfold._core`. |

## Componenti

| Header | Ruolo |
|---|---|
| `conformation.hpp` | Classe `Conformation` (sequenza e posizioni), validità, energia (−1 per ogni contatto H-H non consecutivo), applicazione di una `Move`. |
| `vshd_moves.hpp` | Mosse VSHD: end, corner e crankshaft. |
| `pull_moves.hpp` | Pull moves di Lesh et al. (2003), interne ed estremi. |
| `remc.hpp` | `run_remc`: Replica Exchange Monte Carlo, variante `REMC_m` di Thachuk et al. (2007); `RemcParams`, `RemcResult`, scala delle temperature. |
| `exact.hpp` | `exact_ground_state`: energia minima esatta per enumerazione dei cammini autoevitanti (DFS con branch and bound). |
| `sample_writer.hpp` | `SampleRecorder`: registra in memoria traiettorie e snapshot delle repliche durante la REMC (usato per i dataset). |
| `canonical.hpp` | Forma canonica (turni) e chiave a 64 bit, invarianti per traslazioni, rotazioni e riflessioni. |
| `actions.hpp` | Spazio delle azioni a 6N + 11 slot (residuo, tipo di mossa). |
| `state_graph.hpp` | `StateGraph` e `Oracle`: enumerazione degli stati, grafo delle mosse, distanze BFS dallo stato nativo, percorsi ottimi, pulizia delle traiettorie. |

## Build e collegamento

Non si compila a mano: [`CMakeLists.txt`](../CMakeLists.txt) costruisce la libreria statica `hp_core` da `src/` e la collega al modulo `_core`; lo lancia `pip install -e .` tramite scikit-build-core. Dopo ogni modifica al C++:

```bash
pip install -e ".[dev,ml]"
```

Da Python si usa come `from hpfold import _core` (per esempio `_core.run_remc` in `python/hpfold/simulation.py`). Schede dettagliate di ogni header in [`notes/file_functioning/cpp/include/hp/`](../notes/file_functioning/cpp/include/hp/).
