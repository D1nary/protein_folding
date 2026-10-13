# `python/hpfold/conformation.py`

## 1. Scopo generale del file

È il modulo che contiene tutta la logica della **fase 0** che non riguarda il disegno: lo stato della catena mentre viene costruita, il controllo delle mosse con cui si aggiunge un residuo e la lettura/scrittura del file JSON della conformazione. Non importa Tkinter, quindi può essere testato con pytest senza aprire una finestra: la GUI si limita a disegnare e a chiamare le funzioni definite qui.

Non va confuso con la classe C++ `hp::Conformation` ([conformation.hpp](../../../cpp/include/hp/conformation.hpp)): quella è la struttura dati usata dalla simulazione, con energia e mosse; questo modulo serve solo a costruire, controllare e salvare su file una conformazione dal lato Python.

**Come si inserisce nel progetto:**

- **Usa** solo la libreria standard (`json`, `enum`, `pathlib`).
- **Usato da** [gui.py](../../../python/hpfold/gui.py) (fase 0), che ne prende quasi tutto: `Chain`, `MoveError`, `Coord`, `RESIDUE_TYPES`, `default_coordinates`, `validate_conformation` e `save_conformation`.
- **Usato da** [simulation.py](../../../python/hpfold/simulation.py) (fase 1), che legge la conformazione iniziale con `load_conformation` e salva quella nativa con `save_conformation`, e da [verification.py](../../../python/hpfold/verification.py) (fase `verify`), che rilegge la conformazione nativa con `load_conformation`.
- **Usato dai test**: [test_conformation.py](../../../tests/test_conformation.py) lo verifica direttamente; [test_core.py](../../../tests/test_core.py), [test_simulation.py](../../../tests/test_simulation.py) e [test_gui.py](../../../tests/test_gui.py) usano `validate_conformation`, `are_adjacent` e `default_coordinates` come riferimento indipendente per controllare i risultati del core C++.

Il formato del file JSON è descritto in [initial_conformation.md](../../code_info/initial_conformation.md).

## 2. Componenti

### Tipi e costanti

- **`Coord`** — Alias per una coppia `(x, y)` di interi che identifica una cella della griglia (x = colonna, `0 <= x < width`; y = riga, `0 <= y < height`). È lo stesso significato del `Coord` del C++.
- **`RESIDUE_TYPES`** — La tupla `("H", "P")` con i due tipi di residuo del modello HP. Serve a `set_type` e `validate_conformation` per rifiutare tipi non validi, e alla GUI per creare i pulsanti di scelta del tipo.
- **`MoveError`** — Enumerazione dei motivi per cui un residuo non può essere messo su una cella. La GUI associa a ciascun valore il messaggio mostrato all'utente.

  - **`CHAIN_COMPLETE`** — la catena ha già tutti i residui previsti.
  - **`OUTSIDE_GRID`** — la cella è fuori dalla griglia.
  - **`OCCUPIED`** — la cella è già occupata da un altro residuo.
  - **`NOT_ADJACENT`** — la cella non è adiacente all'ultimo residuo inserito.

### Funzioni libere

- **`are_adjacent(a, b)`** — Dice se due celle sono vicine sul reticolo (su/giù/sinistra/destra, non in diagonale). Serve a `check_move` per verificare che la catena resti connessa; nei test è usata anche per ricalcolare l'energia in Python e confrontarla con quella del C++.
- **`default_coordinates(width, height, length)`** — Restituisce le coordinate di una conformazione di default che entra nella griglia: la catena riempie la griglia riga per riga a serpentina (da sinistra a destra sulle righe pari, da destra a sinistra su quelle dispari), così due residui consecutivi sono sempre adiacenti. Lancia `ValueError` se la catena è più lunga del numero di celle. È usata quando l'utente non disegna la conformazione a mano.
- **`validate_conformation(sequence, coordinates, width, height)`** — Lancia `ValueError` se la conformazione non è valida per la griglia, altrimenti non restituisce nulla. Controlla che ci sia una coordinata per residuo, che la conformazione non sia vuota, che la sequenza contenga solo H e P e che le coordinate formino un cammino auto-evitante dentro la griglia. Quest'ultimo controllo è fatto ripercorrendo il cammino su una `Chain`, così le regole sono le stesse usate mentre si disegna.
- **`save_conformation(path, sequence, coordinates, energy=None)`** — Scrive la conformazione in un file JSON, creando la cartella se non esiste. Se `energy` è indicata (uscita della fase 1) viene aggiunta come terza chiave. Il file è scritto a mano invece che con `json.dump(indent=...)`, che metterebbe ogni numero delle coordinate su una riga a sé.
- **`load_conformation(path)`** — Legge un file scritto da `save_conformation` e restituisce la coppia `(sequence, coordinates)`, con le coordinate riconvertite in tuple di interi.

### Classe `Chain`

Rappresenta la catena **in costruzione** su una griglia `width × height`: le posizioni dei residui inseriti finora, in ordine di catena, e il loro tipo. Un residuo a cui non è ancora stato assegnato il tipo ha tipo `None`. I metodi sono divisi secondo i due passi della GUI.

**Attributi**

- **`width`** / **`height`** — Dimensioni della griglia.
- **`length`** — Numero di residui che la catena deve avere una volta completa.
- **`positions`** — Le celle occupate dai residui inseriti finora, in ordine di catena.
- **`types`** — Il tipo (`"H"`, `"P"` o `None`) di ogni residuo, nello stesso ordine di `positions`.

**Passo 1: forma della catena**

- **`Chain(width, height, length)`** — Costruttore: crea una catena vuota.
- **`contains(cell)`** — Dice se la cella è dentro la griglia.
- **`check_move(cell)`** — Restituisce il motivo (`MoveError`) per cui il prossimo residuo non può andare su `cell`, oppure `None` se può. Il primo residuo può andare su qualunque cella libera della griglia; i successivi devono essere adiacenti all'ultimo inserito.
- **`add(cell)`** — Mette il prossimo residuo su `cell` se la mossa è valida, con tipo `None`. Restituisce `None` se va a buon fine, altrimenti il motivo del rifiuto, lasciando la catena invariata.
- **`undo()`** — Toglie l'ultimo residuo. Restituisce `False` se la catena era vuota.
- **`clear()`** — Toglie tutti i residui.
- **`is_complete`** — Proprietà: vera quando la catena ha esattamente `length` residui. È la condizione con cui la GUI abilita il passaggio al passo 2.

**Passo 2: tipi H/P**

- **`index_at(cell)`** — Indice lungo la catena del residuo che occupa `cell`, oppure `None` se la cella è libera. Serve alla GUI per sapere quale residuo è stato cliccato.
- **`set_type(index, residue_type)`** — Assegna (o cambia) il tipo del residuo `index`. Lancia `ValueError` se il tipo non è H o P.
- **`is_fully_typed`** — Proprietà: vera quando la catena è completa e ogni residuo ha un tipo. È la condizione con cui la GUI abilita il salvataggio.
- **`sequence`** — Proprietà: la sequenza HP in ordine di catena, come stringa. Lancia `ValueError` se la catena non è completa o qualche residuo non ha tipo.

*Nota:* `contains` e `check_move` sono pubblici ma sono chiamati solo all'interno di questo file (`check_move` da `add`, `contains` da `check_move`): la GUI e i test passano sempre da `add`.

*Nota:* `load_conformation` restituisce solo sequenza e coordinate: la chiave `energy`, se presente nel file, viene ignorata. Inoltre non controlla ciò che legge: la validazione va fatta a parte con `validate_conformation`. Nel codice non di test, però, `validate_conformation` è chiamata solo da `write_default_conformation` in [gui.py](../../../python/hpfold/gui.py); in fase 1 il controllo della conformazione letta è lasciato al costruttore C++ di `_core.Conformation`.

## In sintesi

Il file fornisce alla GUI la classe `Chain`, che tiene lo stato della catena mentre l'utente la disegna e rifiuta le mosse non valide spiegandone il motivo, e a tutte le fasi le funzioni per controllare (`validate_conformation`), scrivere (`save_conformation`) e leggere (`load_conformation`) il file JSON con cui la conformazione passa da una fase alla successiva.
