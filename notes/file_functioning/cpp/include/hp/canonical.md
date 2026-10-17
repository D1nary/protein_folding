# `cpp/include/hp/canonical.hpp`

## 1. Scopo generale del file

Definisce la **forma canonica** di una conformazione: la catena descritta dai suoi N − 2 turni (FORWARD, LEFT, RIGHT), presa tra le due immagini speculari con il primo turno diverso da FORWARD a sinistra. Due conformazioni che differiscono solo per una traslazione, una rotazione o una riflessione hanno la stessa forma canonica e quindi la stessa **chiave** a 64 bit. La chiave è il nome di uno stato in tutto il dataset. Vedi [notes/dataset.md](../../../../../notes/dataset.md), §3.

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](../../../../../cpp/include/hp/conformation.hpp).
- **Implementato in** [canonical.cpp](../../../../../cpp/src/canonical.cpp).
- **Usato da** [actions.cpp](../../../../../cpp/src/actions.cpp) (rotazioni), [state_graph.cpp](../../../../../cpp/src/state_graph.cpp) (chiavi dei successori), [sample_writer.cpp](../../../../../cpp/src/sample_writer.cpp) (chiavi degli stati visitati dalla REMC) e [bindings.cpp](../../../../../cpp/bindings/bindings.cpp).

## 2. Componenti

- **`enum Turn`**: `FORWARD = 0`, `LEFT = 1`, `RIGHT = 2`. LEFT trasforma la direzione (dx, dy) in (−dy, dx). Una riflessione scambia LEFT e RIGHT.
- **`MAX_KEY_LENGTH = 42`**: catena più lunga la cui chiave sta in 64 bit (3⁴⁰ < 2⁶⁴).
- **`turns(c)`**, **`mirror_turns(t)`**, **`canonical_turns(c)`**: turni della catena, della sua immagine speculare e della forma canonica.
- **`turns_key(t)`**, **`key_turns(key, n)`**: dai turni alla chiave (numero in base 3, primo turno come cifra più significativa) e ritorno. L'ordine delle chiavi coincide con l'ordine lessicografico dei turni.
- **`canonical_key(c)`**: chiave canonica, calcolata in un solo passaggio senza costruire liste: è il minimo tra la chiave della catena e quella della sua immagine speculare. È la funzione chiamata più spesso nella costruzione del grafo.
- **`free_lattice_side(n)`**: lato (N + 4) del reticolo su cui si ricostruiscono gli stati canonici. La catena è centrata e nessuna mossa raggiunge i bordi, quindi il reticolo si comporta come se fosse infinito.
- **`conformation_from_turns(sequence, t)`**, **`conformation_from_key(sequence, key)`**: ricostruiscono la conformazione (primo legame verso destra, catena centrata). Lanciano `std::invalid_argument` se i turni non danno un cammino auto-evitante.
