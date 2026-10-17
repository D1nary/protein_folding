# `cpp/include/hp/actions.hpp`

## 1. Scopo generale del file

Definisce lo **spazio delle azioni** del dataset (fase 2): un insieme discreto e di dimensione fissa, 6N + 11 slot per una catena di N residui. Ogni slot è una coppia (residuo, tipo di mossa) e contiene, per una data conformazione, al più una mossa. È il "vocabolario" delle uscite del modello di fase 3. La teoria e le ragioni delle scelte sono in [notes/dataset.md](../../../../../notes/dataset.md).

**Come si inserisce nel progetto:**

- **Usa** [vshd_moves.hpp](../../../../../cpp/include/hp/vshd_moves.hpp) e [pull_moves.hpp](../../../../../cpp/include/hp/pull_moves.hpp): non ricalcola la geometria delle mosse, ma classifica quelle che restituiscono. Usa anche [canonical.hpp](../../../../../cpp/include/hp/canonical.hpp) per le rotazioni `turn_left` / `turn_right`.
- **Implementato in** [actions.cpp](../../../../../cpp/src/actions.cpp).
- **Usato da** [state_graph.cpp](../../../../../cpp/src/state_graph.cpp), che costruisce i successori di ogni stato, e da [bindings.cpp](../../../../../cpp/bindings/bindings.cpp).

## 2. Componenti

### Enum `MoveType`

I 18 tipi di mossa, tutti definiti rispetto alla catena (L e R sono rotazioni di 90° rispetto a un legame):

- `END_L`, `END_R`, `END_B`: end move VSHD;
- `CORNER`: corner move, che coincide con la pull move che termina subito;
- `CRANKSHAFT`: crankshaft della coppia (k, k+1);
- `PULL_NEXT_L/R`, `PULL_PREV_L/R`: pull move di un residuo interno;
- `PULL_END_FF` … `PULL_END_RR`: le 9 pull move di un estremo.

### Funzioni

- **`action_count(n)`**, **`action_table(n)`**, **`action_index(n, residue, type)`**: dimensione, elenco e numerazione degli slot (residuo per residuo, e per ogni residuo nell'ordine di `MoveType`).
- **`move_type_name(type)`**, **`mirror_move_type(type)`**: nome di un tipo e tipo dell'immagine speculare (L↔R).
- **`mirror_permutation(n)`**: permutazione degli slot sotto una riflessione. È un'involuzione e serve all'augmentation per simmetria.
- **`classify_move(c, k, move, pull)`**: slot di una mossa restituita per il residuo `k` da `vshd_moves` (`pull = false`) o da `pull_moves` (`pull = true`). Il flag serve perché un crankshaft e una pull move che sposta solo due residui possono avere la stessa forma.
- **`action_moves(c)`**: tutte le mosse possibili della conformazione, una per slot; una mossa vuota vuol dire slot non valido. Lancia `std::logic_error` se due mosse diverse finiscono nello stesso slot, e garantisce così la corrispondenza 1:1 con il move set.
