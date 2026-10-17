# `cpp/include/hp/state_graph.hpp`

## 1. Scopo generale del file

Contiene l'**oracolo** che etichetta il dataset (fase 2): il grafo delle conformazioni collegate da una mossa, le distanze dallo stato nativo, le mosse ottime, i percorsi verso lo stato nativo e la pulizia delle traiettorie REMC. Vedi [notes/dataset.md](../../../../../notes/dataset.md), §4 e §5.

**Come si inserisce nel progetto:**

- **Usa** [actions.hpp](../../../../../cpp/include/hp/actions.hpp) per le mosse di ogni stato e [canonical.hpp](../../../../../cpp/include/hp/canonical.hpp) per le chiavi.
- **Implementato in** [state_graph.cpp](../../../../../cpp/src/state_graph.cpp).
- **Usato da** [bindings.cpp](../../../../../cpp/bindings/bindings.cpp), e da lì da [dataset.py](../../../../../python/hpfold/dataset.py).

## 2. Componenti

### Costanti

- **`INVALID_ACTION = -1`**: valore di `next_distance` per uno slot non valido.
- **`UNKNOWN_DISTANCE = -2`**: distanza non nota. Lo stato non è nel grafo, oppure dal grafo non si raggiunge lo stato nativo.

### Funzioni libere

- **`successors(n, key)`**: per ogni azione possibile, lo slot e la chiave dello stato di arrivo.
- **`enumerate_states(n)`**: tutte le conformazioni canoniche di n residui, in ordine di chiave. Usa una ricerca in profondità con la stessa rottura di simmetria di `exact.cpp`: per N = 16 sono 802 075 stati.
- **`prune_trajectory(trajectory, oracle)`**: taglia la traiettoria di un walker al primo stato nativo e restituisce un `PrunedTrajectory`, che contiene:
  - le lunghezze grezza, dopo la loop truncation e del cammino minimo tra gli stati visitati (con le scorciatoie);
  - quel cammino minimo.

### Classe `StateGraph`

Grafo orientato su un insieme di stati: c'è un arco u → v quando un'azione di u porta in v. Gli archi verso stati fuori dall'insieme vengono scartati.

- La costruzione divide i nodi tra più thread.
- Gli archi sono salvati **invertiti** (predecessori, in righe compresse), perché la BFS parte dagli stati nativi.
- Il grafo non è simmetrico: alcune pull moves di un estremo non hanno l'inversa in una mossa sola.

Metodi:

- **`index(key)`**: nodo di una chiave (ricerca binaria sulle chiavi ordinate), oppure −1.
- **`distances_to(targets)`**: BFS dai nodi bersaglio.

### Classe `Oracle`

Collega un grafo a una sequenza.

- **Costruttore:** calcola l'energia di ogni nodo. Gli stati nativi sono tutti i nodi con l'energia bersaglio (la minima del grafo, se non viene indicata); poi calcola le distanze.
- **`label(keys)`**: per ogni stato restituisce energia, distanza, maschera, `next_distance` e `next_energy_delta` per tutti gli slot (struct `Labels`). Uno stato fuori dal grafo ha distanza 1 + la minima distanza nota dei suoi successori.
- **`path(start, seed)`**: percorso fino a uno stato nativo, scegliendo a ogni passo una delle mosse ottime con un generatore inizializzato da `seed`.
