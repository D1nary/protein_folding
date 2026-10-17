# `cpp/include/hp/sample_writer.hpp`

## 1. Scopo generale del file

Registra le configurazioni visitate dalla REMC per costruire il dataset (fase 2). Nonostante il nome storico del file, il C++ non scrive su disco: i dati restano in memoria, Python li legge attraverso il wrapper e scrive i file. È la stessa regola per cui il C++ non legge né il TOML né il JSON. Vedi [notes/dataset.md](../../../../../notes/dataset.md), §5.

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](../../../../../cpp/include/hp/conformation.hpp) e, nell'implementazione, [canonical.hpp](../../../../../cpp/include/hp/canonical.hpp) per le chiavi.
- **Implementato in** [sample_writer.cpp](../../../../../cpp/src/sample_writer.cpp).
- **Usato da** [remc.cpp](../../../../../cpp/src/remc.cpp): `run_remc` riceve un puntatore opzionale a `SampleRecorder`. Python lo crea in [dataset.py](../../../../../python/hpfold/dataset.py) (`sample_remc`).

## 2. Componenti

### Struct `Snapshot`

Stato di una replica alla fine di un ciclo: `cycle`, `replica` (indice di temperatura, 0 = la più fredda), `walker`, `energy`, `key`.

### Classe `SampleRecorder`

- **Costruttore `SampleRecorder(max_steps, save_every)`**: lancia `std::invalid_argument` se un parametro non è positivo.
- **Traiettorie dei walker.** Un walker è una conformazione seguita attraverso gli scambi: uno scambio cambia la sua temperatura, non la sua conformazione. Per questo la traiettoria di un walker è una catena di mosse singole, che si può ripulire. Il recorder salva la chiave dello stato iniziale e quella dopo ogni mossa accettata che cambia lo stato, fino a `max_steps` mosse per walker. Una mossa può trasformare la catena nella propria immagine speculare: per la chiave è lo stesso stato e non viene salvato.
- **Snapshot.** Ogni `save_every` cicli salva lo stato di ogni replica. Sono i decoy del dataset alle varie temperature.
- **Metodi chiamati da `run_remc`:** `start`, `record_move`, `wants_snapshot`, `record_snapshot`. Non usano il generatore casuale, quindi la simulazione è identica con o senza recorder.
