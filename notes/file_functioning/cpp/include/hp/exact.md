# `cpp/include/hp/exact.hpp`

## 1. Scopo generale del file

È l'header che dichiara il calcolo dello **stato fondamentale esatto** di una sequenza HP corta: l'energia minima viene trovata per **enumerazione esaustiva** di tutti i cammini auto-evitanti che entrano nel reticolo, con una ricerca in profondità (depth-first search) accelerata da un branch and bound. Non fa parte della simulazione: serve a verificare che il minimo trovato dal REMC sia davvero lo stato fondamentale.

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](../../../../../cpp/include/hp/conformation.hpp), da cui prende `Conformation`, e `<string>`.
- **Implementato in** [exact.cpp](../../../../../cpp/src/exact.cpp).
- **Usato da** [bindings.cpp](../../../../../cpp/bindings/bindings.cpp), che espone tutto a Python. Da lì lo chiama [verification.py](../../../../../python/hpfold/verification.py) (fase `verify`), che confronta l'energia della conformazione salvata dal REMC con il minimo esatto, e i test in [test_simulation.py](../../../../../tests/test_simulation.py).

Tutto è dentro il namespace `hp`.

## 2. Componenti

### Struct `ExactResult`

Raccoglie ciò che l'enumerazione restituisce al chiamante.

- **`best`** — Una delle conformazioni a energia minima (possono essercene più d'una: viene restituita la prima trovata).
- **`energy`** — L'energia minima esatta della sequenza su quel reticolo.
- **`explored_nodes`** — Numero di catene parziali visitate dalla ricerca; serve a valutare quanto è costata l'enumerazione e quanto taglia il bound.

### Funzioni libere

- **`exact_ground_state(const std::string& sequence, int width, int height)`** — Trova l'energia minima di `sequence` su tutti i cammini auto-evitanti che entrano in un reticolo `width × height` e restituisce un `ExactResult`. Lancia `std::invalid_argument` se la sequenza è vuota o non è fatta solo di H e P, se il reticolo non ha dimensioni positive o se nessun cammino di quella lunghezza entra nel reticolo.

  La catena viene costruita un residuo alla volta, provando i nodi adiacenti liberi e tornando indietro quando non ce ne sono più. Due accorgimenti tengono piccola la ricerca:

  - **Simmetria** — il primo legame è fissato verso destra e la prima svolta verso l'alto, così di ogni cammino viene costruita una sola delle 8 copie ruotate/specchiate.
  - **Bound** — una catena parziale viene abbandonata quando, anche formando il massimo numero di contatti ancora possibili per i residui mancanti (2 per ogni H, 3 per l'ultimo residuo), non può battere la migliore conformazione trovata finora.

  Nota: il costo cresce comunque in modo esponenziale con la lunghezza della catena, quindi la funzione è pensata per sequenze corte. Per questo [verification.py](../../../../../python/hpfold/verification.py) la salta se la catena è più lunga di `exact.max_length` (in [default.toml](../../../../../config/default.toml)).
