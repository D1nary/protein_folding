# File della conformazione iniziale (fase 0 → fase 1)

La GUI di fase 0 scrive la conformazione iniziale nel percorso indicato da `protein.initial_conformation` (di default `data/input/initial_conformation.json`). È l'unico dato che passa dalla fase 0 alla fase 1.

## Formato

Un oggetto JSON con due chiavi:

```json
{
  "sequence": "HPPH",
  "coordinates": [[0, 0], [1, 0], [1, 1], [0, 1]]
}
```

- **`sequence`**: stringa di soli `H` e `P`, lunga N (`protein.length`). Il carattere `i` è il tipo del residuo `i`.
- **`coordinates`**: lista di N coppie `[x, y]` di interi, **in ordine di catena**: la coppia `i` è la posizione del residuo `i`, quindi `coordinates[i]` e `sequence[i]` descrivono lo stesso residuo.

## Convenzione sulle coordinate

- `x` è la colonna, `0 <= x < protein.length + 4`.
- `y` è la riga, `0 <= y < protein.length + 4`.
- L'origine `(0, 0)` è il nodo in alto a sinistra nella finestra, e `y` cresce verso il basso. Il verso dell'asse non ha effetti sull'energia, che dipende solo da quali residui sono adiacenti.

Le dimensioni della griglia non sono nel file.

La GUI disegna su una griglia quadrata di lato `protein.length + 4` (`grid_side` in [gui.py](../../python/hpfold/gui.py)). È lo stesso reticolo su cui il dataset ricostruisce ogni stato (`free_lattice_side`, vedi [notes/dataset.md](../dataset.md)). La fase 1 rilegge poi il file sulla griglia di `[lattice]`, che quindi deve essere almeno di lato `protein.length + 4` (30 × 30 di default).

## Cosa è garantito

Un file scritto dalla GUI rispetta sempre questi vincoli:

- `len(sequence) == len(coordinates) == protein.length`;
- tutte le coordinate sono dentro la griglia;
- nessuna cella è occupata due volte (catena self-avoiding);
- due residui consecutivi sono su celle adiacenti (su/giù/sinistra/destra).

Il file però può essere modificato a mano o diventare incoerente se si cambia la griglia nel TOML: chi lo legge deve ricontrollarlo con `validate_conformation`.

## Come si legge

In Python, in [conformation.py](../../python/hpfold/conformation.py):

```python
from hpfold.conformation import load_conformation, validate_conformation

sequence, coordinates = load_conformation(config["protein"]["initial_conformation"])
validate_conformation(sequence, coordinates, width, height)
```

`sequence` è una `str` e `coordinates` una `list[tuple[int, int]]`.

## Come arriva al C++

Il C++ non legge il JSON, per lo stesso motivo per cui non legge il TOML: non serve una libreria di parsing. In fase 1 `simulation.py` legge il file e passa i due valori al wrapper; pybind11 li converte da solo nei tipi STL corrispondenti:

| Python | C++ |
|---|---|
| `str` | `std::string` |
| `list[tuple[int, int]]` | `std::vector<std::pair<int, int>>` |

Il formato è stato scelto proprio perché coincide con quello che serve al costruttore di `hp::Conformation`, ad esempio:

```cpp
Conformation(std::string sequence, std::vector<std::pair<int, int>> coordinates,
             int width, int height);
```

## Le due origini della sequenza

`protein.sequence` nel TOML e `sequence` nel JSON descrivono lo stesso dato. La regola è: **la fase 1 legge sempre e solo il JSON**. All'avvio la GUI chiede se si vuole disegnare una nuova proteina:

- **Yes**: la forma e i tipi H/P vengono dal disegno; `protein.sequence` non viene usata.
- **No**: nel JSON viene scritta la proteina di default, cioè `protein.sequence` con una forma generata automaticamente (`default_coordinates`: la catena riempie la griglia a serpentina, riga per riga).
- **Cancel**: non viene scritto nulla.

In entrambi i casi `protein.length` fissa il numero di residui.
