- **`hpfold`**: è il pacchetto Python del progetto, cioè quello che usi con `import hpfold` o `python -m hpfold --config <file> <fase>`. I suoi sorgenti stanno in `python/hpfold/` e orchestra tutta la pipeline (GUI, simulazione, dataset, ML). È anche il nome del progetto in pyproject.toml e in CMakeLists.txt.
- **`_core`**: è il modulo Python compilato (un file `.so`) che rende il codice C++ chiamabile da Python. Viene costruito con pybind11 a partire da bindings.cpp e installato dentro il pacchetto, quindi si importa come `hpfold._core`. Contiene al suo interno `hp_core`; lo usa ad esempio simulation.py per lanciare la REMC.
- **`hp_core`**: è la libreria statica C++ (`libhp_core.a`) con la logica vera della simulazione, compilata dai sorgenti in `cpp/src/` con gli header in `cpp/include/`. Non sa nulla di Python e non viene installata: è un prodotto intermedio che viene collegato dentro `_core`.

In sintesi, la catena è `hp_core` (C++ puro) → `_core` (wrapper che lo espone a Python) → `hpfold` (pacchetto Python che lo usa).


