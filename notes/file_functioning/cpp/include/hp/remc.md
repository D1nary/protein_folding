# `cpp/include/hp/remc.hpp`

## 1. Scopo generale del file

È l'header che dichiara l'algoritmo di ottimizzazione vero e proprio: il **Replica Exchange Monte Carlo** (Thachuk et al., 2007), nella variante `REMC_m`, in cui la ricerca locale mescola pull moves e mosse VSHD. È il punto d'ingresso del core C++: riceve una conformazione iniziale e i parametri, restituisce la migliore conformazione trovata. L'header rimanda a [notes/samplings.md](https://claude.ai/epitaxy/notes/samplings.md) per la teoria.

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](https://claude.ai/epitaxy/cpp/include/hp/conformation.hpp) per `Conformation`, più `<cstdint>` e `<vector>`.
- **Implementato in** [remc.cpp](https://claude.ai/epitaxy/cpp/src/remc.cpp), che è l'unico punto in cui vengono chiamati `pull_moves` e `vshd_moves` nel codice C++.
- **Usato da** [bindings.cpp](https://claude.ai/epitaxy/cpp/bindings/bindings.cpp), che espone tutto a Python. Da lì lo chiama [simulation.py](https://claude.ai/epitaxy/python/hpfold/simulation.py), che costruisce i parametri dal file di configurazione e lancia la simulazione.

Tutto è dentro il namespace `hp`.

## 2. Componenti

### Struct `RemcParams`

Raccoglie i parametri della simulazione. Viene riempita da Python con i valori del file di configurazione; di proposito non ha valori di default.

- **`n_replicas`** — Numero di repliche (M), almeno 2.
- **`t_min`** — Temperatura della replica più fredda (T₁), con k_B = 1.
- **`t_max`** — Temperatura della replica più calda (T_M).
- **`local_steps`** — Numero di passi Monte Carlo locali (φ) tra due tentativi di scambio.
- **`n_cycles`** — Numero di cicli, dove un ciclo è campionamento locale più scambi.
- **`rho`** — Probabilità che un passo locale usi una pull move invece di una mossa VSHD.
- **`seed`** — Seme del generatore di numeri casuali, per rendere la simulazione riproducibile.

### Struct `RemcResult`

Raccoglie ciò che la simulazione restituisce al chiamante.

- **`best`** — La conformazione a energia più bassa visitata durante tutta la simulazione.
- **`best_energy`** — L'energia di `best`.
- **`temperatures`** — Le temperature delle repliche, in ordine crescente.
- **`final_energies`** — L'energia di ciascuna replica al termine della simulazione.
- **`swap_attempts`** — Numero di scambi tra repliche proposti.
- **`swap_accepted`** — Numero di scambi accettati; insieme a `swap_attempts` serve a valutare se la scala di temperature è ben scelta.

### Funzioni libere

- **`temperature_ladder(int n_replicas, double t_min, double t_max)`** — Restituisce le temperature delle repliche, spaziate linearmente tra `t_min` e `t_max`. Serve a `run_remc` per assegnare una temperatura a ogni replica.
- **`run_remc(const Conformation& initial, const RemcParams& params)`** — Esegue l'intera simulazione REMC facendo partire tutte le repliche da `initial` e restituisce un `RemcResult`. Lancia `std::invalid_argument` se i parametri non sono validi.


