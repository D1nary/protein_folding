# `cpp/include/hp/vshd_moves.hpp`

## 1. Scopo generale del file

È l'header che dichiara il generatore delle **mosse VSHD** (Verdier-Stockmayer, Hilhorst-Deutch), uno dei due insiemi di mosse locali con cui la simulazione modifica una conformazione; l'altro è quello delle pull moves. Il riferimento citato nel file è Thachuk et al. (2007).

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](https://claude.ai/epitaxy/cpp/include/hp/conformation.hpp), da cui prende `Conformation` e `Move`, e `<vector>`.
- **Implementato in** [vshd_moves.cpp](https://claude.ai/epitaxy/cpp/src/vshd_moves.cpp).
- **Usato da** [remc.cpp](https://claude.ai/epitaxy/cpp/src/remc.cpp), che a ogni passo di ricerca locale sceglie tra pull moves e mosse VSHD, e da [bindings.cpp](https://claude.ai/epitaxy/cpp/bindings/bindings.cpp), che espone la funzione a Python (usata lì solo dai test in [test_core.py](https://claude.ai/epitaxy/tests/test_core.py)).

Tutto è dentro il namespace `hp`.

## 2. Componenti

Il file dichiara una sola componente.

- **`vshd_moves(const Conformation& conformation, int k)`** — Restituisce tutte le mosse VSHD ammissibili per il residuo `k`, cioè quelle i cui nodi di destinazione sono liberi e dentro il reticolo. Viene chiamata da REMC per ottenere le mosse candidate tra cui sceglierne una da passare a `Conformation::apply` Successivamente, REMC seglierà pull moves o vshd moves con probabilità rho. 

  Le mosse che può restituire sono di tre tipi:

  - **End move** — sposta un residuo che si trova a un'estremità della catena.
  - **Corner move** — sposta un residuo interno che forma un angolo con i suoi due vicini.
  - **Crankshaft move** — ribalta insieme i due residui centrali di una piega a U.

  Nota: tra le vshd moves possibili, viene proposta solo quella adatta a quel determinato residuo `k`. 


