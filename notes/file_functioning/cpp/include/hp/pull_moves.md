# `cpp/include/hp/pull_moves.hpp`

## 1. Scopo generale del file

È l'header che dichiara il generatore delle **pull moves** di Lesh, Mitzenmacher e Whitesides (2003), il secondo insieme di mosse locali del progetto accanto alle mosse VSHD. A differenza di queste, una pull move può trascinare con sé un tratto intero della catena, quindi una singola mossa può spostare molti residui. L'header rimanda a [notes/move_set.md](https://claude.ai/epitaxy/notes/move_set.md) per la descrizione teorica.

**Come si inserisce nel progetto:**

- **Usa** [conformation.hpp](https://claude.ai/epitaxy/cpp/include/hp/conformation.hpp), da cui prende `Conformation` e `Move`, e `<vector>`.
- **Implementato in** [pull_moves.cpp](https://claude.ai/epitaxy/cpp/src/pull_moves.cpp).
- **Usato da** [remc.cpp](https://claude.ai/epitaxy/cpp/src/remc.cpp), che a ogni passo di ricerca locale sceglie tra pull moves e mosse VSHD, e da [bindings.cpp](https://claude.ai/epitaxy/cpp/bindings/bindings.cpp), che espone la funzione a Python (usata lì solo dai test in [test_core.py](https://claude.ai/epitaxy/tests/test_core.py)).

Tutto è dentro il namespace `hp`.

## 2. Componenti

Il file dichiara una sola componente.

- **`pull_moves(const Conformation& conformation, int k)`** — Restituisce tutte le pull moves ammissibili per il residuo `k`, cioè quelle i cui nodi di destinazione sono liberi e dentro il reticolo. Viene chiamata da REMC per ottenere le mosse candidate tra cui sceglierne una da passare a `Conformation::apply`.

  Copre due casi:

  - **Residuo interno alla catena** — il residuo si sposta su un nodo diagonale e, se serve, il resto della catena viene tirato dietro di lui; la catena può essere tirata verso entrambe le estremità.
  - **Primo o ultimo residuo** — l'estremità avanza di due nodi in una zona libera e il resto della catena la segue.
