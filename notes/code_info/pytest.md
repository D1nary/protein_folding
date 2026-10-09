**pytest** è il framework di test standard per Python: tu scrivi funzioni il cui nome inizia con `test_` e dentro usi dei normali `assert`; pytest le trova da solo, le esegue una per una e ti dice quali passano e quali falliscono (mostrando i valori che hanno fatto fallire l'`assert`).

Non ho eseguito i test: quanto segue viene dalla lettura dei file.

## Come è collegato al progetto

- **Installazione**: in [pyproject.toml](https://claude.ai/epitaxy/pyproject.toml) pytest è nel gruppo opzionale `dev`, quindi arriva solo con `pip install -e ".[dev]"`. Chi installa solo il pacchetto `hpfold` non se lo porta dietro.
- **Configurazione**: la sezione `[tool.pytest.ini_options]` contiene solo `testpaths = ["tests"]`, cioè lanciando `pytest` senza argomenti cerca i test solo in `tests/`.
- **Scoperta dei test**: dentro `tests/` pytest raccoglie i file `test_*.py` e, al loro interno, le funzioni `test_*`. Non serve registrarle da nessuna parte.

## Cosa viene testato oggi

C'è un solo file, [tests/test_smoke.py](https://claude.ai/epitaxy/tests/test_smoke.py), con due controlli minimi ("smoke test", cioè verificano solo che l'installazione stia in piedi):

2. **`test_core_extension_is_importable`**: importa `hpfold._core`, l'estensione C++ compilata con pybind11, e verifica che esponga `run_remc`. Se la compilazione è fallita o il binding non è stato registrato, fallisce qui.
3. **`test_default_config_loads`**: carica `config/default.toml` con `load_config` e verifica che ci siano le sezioni `lattice`, `protein`, `remc`, `output` e che `protein.length` coincida con la lunghezza della sequenza.

Non ci sono ancora test sulla correttezza dell'algoritmo (energia, pull moves, scambi REMC).

## Come si usa

Dalla radice del progetto, con l'ambiente virtuale attivo:

```bash
pytest
```

Varianti utili sono `pytest -v` (elenca ogni test con l'esito) e `pytest tests/test_smoke.py::test_default_config_loads` (ne esegue uno solo).

I test importano il pacchetto **installato**, non i sorgenti: dopo una modifica al C++ devi rieseguire `pip install -e ".[dev]"` prima di `pytest`, altrimenti testi la versione compilata precedente. Le modifiche ai file Python invece sono subito attive.
