# `tests/`

Test automatici del progetto, eseguiti con `pytest` (configurato in `pyproject.toml` con `testpaths = ["tests"]`).

| File | Cosa verifica |
|---|---|
| `test_smoke.py` | L'estensione C++ si importa e la configurazione di default si legge. |
| `test_core.py` | Core C++: energia, mosse VSHD e pull moves su casi costruiti a mano, validità delle mosse lungo una camminata casuale. |
| `test_conformation.py` | Logica della catena: mosse valide e non valide, annullamento, tipi H/P, file JSON. |
| `test_gui.py` | Parti di `gui.py` che non aprono finestre: conversione pixel/nodo, proteina di default, contatti e fotogrammi dell'animazione. |
| `test_simulation.py` | REMC: scala delle temperature, parametri non validi, riproducibilità con lo stesso `seed`, energia minima, enumerazione esatta, fase `verify`. |
| `test_policy_dataset.py` | Spazio delle azioni, chiave canonica, simmetrie, distanze dell'oracolo, file del policy dataset. |
| `test_discriminator_dataset.py` | Energie, trappole cinetiche, fasce, bilanciamento 1:1, duplicati, split per sequenza, riproducibilità. |
| `test_discriminator_model.py` | Feature, assenza di campi che rivelano la risposta, invarianza per simmetrie, AUROC, salvataggio e rilettura del modello, riproducibilità. Saltato se `torch` non è installato. |
| `test_discriminator_score.py` | Fase `discriminator_score`: lettura e controllo della conformazione in ingresso, uscite del giudizio, fase da riga di comando. Saltato se mancano `torch` o `tkinter`. |
| `test_folding_path.py` | Percorso verso lo stato nativo: ogni mossa riduce la distanza di uno, termina in un nativo, è deterministico. |

## Uso

Dalla radice del progetto, con le dipendenze `dev` (e `ml` per i test del modello) installate:

```bash
pytest
```

```bash
pytest tests/test_core.py -v
```
