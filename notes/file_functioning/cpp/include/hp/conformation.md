# `cpp/include/hp/conformation.hpp`

## 1. Scopo generale del file

È l'header che dichiara la struttura dati centrale del progetto: una **conformazione** del modello HP su reticolo 2D, cioè una catena di residui H/P con la loro posizione sul reticolo e la relativa energia. Definisce anche i tipi di base (coordinate, mosse) e alcune funzioni geometriche sul reticolo.

**Come si inserisce nel progetto:**

- **Usa** solo la libreria standard (`<string>`, `<utility>`, `<vector>`).
- **Implementato in** [conformation.cpp](https://claude.ai/epitaxy/cpp/src/conformation.cpp).
- **Usato da** [vshd_moves.hpp](https://claude.ai/epitaxy/cpp/include/hp/vshd_moves.hpp) e [pull_moves.hpp](https://claude.ai/epitaxy/cpp/include/hp/pull_moves.hpp) (generazione delle mosse), [remc.hpp](https://claude.ai/epitaxy/cpp/include/hp/remc.hpp) (simulazione Replica Exchange Monte Carlo) e [bindings.cpp](https://claude.ai/epitaxy/cpp/bindings/bindings.cpp) (esposizione a Python).

Tutto è dentro il namespace `hp`.

## 2. Componenti

### Tipi

- **`Coord`** — Alias per una coppia `(x, y)` che identifica un nodo del reticolo (x = colonna, y = riga). È il tipo usato ovunque per indicare una posizione.
- **`Move`** — Alias per una lista di coppie (indice del residuo, nuova posizione). Rappresenta una mossa come l'insieme dei residui che si spostano; è ciò che producono i generatori di mosse e che `apply` consuma.

### Funzioni libere

- **`are_adjacent(Coord a, Coord b)`** — Dice se due nodi sono collegati da un lato del reticolo. Serve per verificare la connessione della catena e i contatti tra residui.
- **`are_diagonal(Coord a, Coord b)`** — Dice se due nodi sono angoli opposti di un quadrato unitario. Serve ai generatori di mosse (mosse d'angolo e pull moves).
- **`neighbours(Coord c)`** — Restituisce i quattro nodi adiacenti a `c`, senza controllare che siano dentro il reticolo. Serve per esplorare l'intorno di un residuo.

### Classe `Conformation`

Rappresenta una catena di residui H/P posizionata su un reticolo di dimensioni `width × height`.

**Metodi pubblici**

- **`Conformation(sequence, coordinates, width, height)`** — Costruttore: crea la conformazione e lancia `std::invalid_argument` se non è valida (sequenza non H/P, coordinate mancanti, residui fuori reticolo, sovrapposti o non connessi).
- **`size()`** — Numero di residui della catena.
- **`width()`** / **`height()`** — Dimensioni del reticolo.
- **`sequence()`** — La sequenza H/P in sola lettura.
- **`coordinates()`** — Le posizioni di tutti i residui in sola lettura, in ordine di catena.
- **`position(int i)`** — Posizione del residuo `i`; usato intensamente dai generatori di mosse.
- **`in_bounds(Coord c)`** — Dice se il nodo è dentro il reticolo. *Nota:* pur essendo pubblico, è chiamato solo all'interno di [conformation.cpp](https://claude.ai/epitaxy/cpp/src/conformation.cpp).
- **`residue_at(Coord c)`** — Indice del residuo che occupa il nodo, oppure `-1` se è vuoto o fuori reticolo.
- **`is_free(Coord c)`** — Dice se il nodo è dentro il reticolo e non occupato: è la domanda che i generatori di mosse fanno per sapere dove un residuo può andare.
- **`is_valid()`** — Verifica che la conformazione sia un cammino auto-evitante dentro il reticolo. *Nota:* nel codice C++ è usato solo dal costruttore; all'esterno è chiamato soltanto dai test Python tramite i binding.
- **`energy()`** — Energia del modello HP: `-1` per ogni coppia di residui H adiacenti sul reticolo ma non consecutivi nella catena. È la grandezza che REMC minimizza.
- **`apply(const Move& move)`** — Applica una mossa e restituisce la mossa inversa, così da poterla annullare se viene rifiutata. Non controlla la mossa (non verifica che le nuove posizioni siano dentro il reticolo, che siano libere, né che la catena resti connessa) piochè il controllo è stato già fatto da `vshd_moves()` o `pull_moves()`.

**Membri privati**

- **`cell(Coord c)`** — Helper che converte una coordinata nell'indice lineare usato per `occupancy_`.
- **`sequence_`** — La sequenza H/P.
- **`coordinates_`** — La posizione di ogni residuo.
- **`width_`** / **`height_`** — Dimensioni del reticolo.
- **`occupancy_`** — Per ogni nodo del reticolo, l'indice del residuo che lo occupa o `-1`. Esiste per rispondere in tempo costante a "questo nodo è libero?".
