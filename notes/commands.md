# Comandi da terminale

Come eseguire dal terminale le diverse parti del progetto. Ogni parte è una
*fase* del pacchetto `hpfold`, lanciata sempre nello stesso modo:

```bash
python -m hpfold --config config/default.toml <fase>
```

- `--config` è il file TOML con tutti i parametri. Per un esperimento diverso
  si copia `config/default.toml` e si passa la copia.
- I comandi vanno lanciati dalla **radice del progetto**: i percorsi relativi
  del TOML (`data/...`, `config/...`) sono risolti rispetto a lì.
- `python` è quello dell'ambiente virtuale: prima `source .venv/bin/activate`,
  oppure si scrive `.venv/bin/python` al posto di `python`. Il `python` di
  sistema non ha il pacchetto installato.

---

## 0. Preparazione (una volta sola)

```bash
sudo apt install build-essential cmake python3-dev python3-venv python3-tk
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ml]"
```

`pip install -e` compila il C++ (modulo `hpfold._core`) e installa il pacchetto.
Dopo una modifica al C++ va rieseguito; le modifiche ai file Python sono invece
subito attive. Il gruppo `ml` installa PyTorch, che serve solo al modello del
discriminator (§3 e §4).

Per controllare che tutto funzioni:

```bash
pytest
```

---

## 1. GUI: conformazione iniziale

```bash
python -m hpfold --config config/default.toml gui
```

Una prima finestra chiede **"Do you want to draw a new protein?"**:

- **Yes**: si apre la griglia. Si disegna la catena cliccando nodo dopo nodo
  (step 1), poi si assegnano i tipi H/P (step 2) e si preme **Save**.
- **No**: si salva la proteina di default, `protein.sequence` con una forma a
  serpentina.
- **Cancel**: esce senza scrivere nulla.

| | |
|---|---|
| Parametri | `[protein]` (`length`, `sequence`), `[gui]` (colori e dimensioni) |
| Output | `data/input/initial_conformation.json` (`protein.initial_conformation`) |

Il file prodotto è l'input della REMC. Dettagli in
[code_info/initial_conformation.md](code_info/initial_conformation.md).

---

## 2. REMC: ricerca della conformazione nativa

```bash
python -m hpfold --config config/default.toml simulate
```

Parte dalla conformazione della GUI ed esegue il Replica Exchange Monte Carlo
sul reticolo `[lattice]`. A fine esecuzione stampa sequenza, energia minima e
scambi accettati.

| | |
|---|---|
| Input | `data/input/initial_conformation.json` (quindi va eseguita prima la GUI) |
| Parametri | `[lattice]`, `[remc]` (repliche, temperature, cicli, `seed`, ...) |
| Output | `data/output/native_conformation.json` (`output.native_conformation`), con la chiave `energy` |

### Verifica (facoltativa)

```bash
python -m hpfold --config config/default.toml verify
```

Calcola il vero minimo di energia per enumerazione esatta e dice se la REMC
l'ha raggiunto. Funziona solo per catene lunghe al massimo `exact.max_length`
(25), perché il costo cresce in modo esponenziale con la lunghezza.

---

## 3. Creazione del dataset del discriminator

```bash
python -m hpfold --config config/default.toml discriminator_dataset
```

Genera `n_sequences` sequenze casuali di `length` residui. Per ciascuna lancia
una REMC propria e raccoglie conformazioni native e decoy (1:1), divise in
fasce di difficoltà (`near`, `mid`, `far`) e trappole cinetiche, poi le divide
in train, val e test per sequenza. A fine esecuzione stampa il riepilogo.

Non usa la GUI né l'output della fase `simulate`: è indipendente. Con i
parametri di default (2000 sequenze di 16 residui) richiede circa 30 minuti.

| | |
|---|---|
| Parametri | `[discriminator_dataset]` e `[discriminator_dataset.remc]` |
| Output | `data/output/datasets/discriminator_dataset/` (`output.discriminator_dataset_dir`) |

File prodotti: `train.csv`, `val.csv`, `test.csv`, `sequences.csv`,
`provenance.csv`, `meta.json`, `report.txt`. Dettagli in
[discriminator_dataset.md](discriminator_dataset.md).

---

## 4. Modello ML discriminator

### 4.1 Addestramento

```bash
python -m hpfold --config config/default.toml discriminator_train
```

Addestra il modello principale (GNN sulla conformazione + CNN 1D sulla
sequenza) e le baseline `B0a`, `B0b`, `B1b`, li valuta su val e test e stampa
la tabella di confronto.

| | |
|---|---|
| Input | il dataset del §3 (quindi va eseguito prima `discriminator_dataset`) |
| Parametri | `[discriminator_model]` (dimensioni, epoche, learning rate, baseline, ...) |
| Output | `data/output/models/discriminator/` (`output.discriminator_model_dir`) |

File prodotti: `discriminator.pt` (pesi del modello), `report.txt`,
`metrics.json`, `history.csv`, `test_predictions.csv`.

### 4.2 Giudizio di una conformazione

```bash
python -m hpfold --config config/default.toml discriminator_score
```

Usa il modello già addestrato per dire se una conformazione è nativa. Una prima
finestra chiede **"Do you want to draw the conformation to judge?"**:

- **Yes**: si apre lo stesso editor della GUI (§1), con una catena lunga quanto
  quelle del dataset (16 residui). Si preme **Evaluate** per avere il giudizio;
  la finestra resta aperta e si possono provare altre conformazioni.
- **No**: si usa la conformazione di
  [config/discriminator_input.toml](../config/discriminator_input.toml)
  (`discriminator_model.input_conformation`). Per cambiarla si modificano
  `sequence` e `coordinates` in quel file.
- **Cancel**: esce senza fare nulla.

Il risultato compare in una finestra ed è stampato anche sul terminale:

```
Sequence: HHHPPPPHHHHPHHHH
Verdict: native
Probability of native: 1.0000
Logit: 11.087
Energy (predicted): -7.20   exact: -7
Lowest energy of the sequence (predicted): -6.78
Gap from the lowest energy (predicted): 0.01
```

| | |
|---|---|
| Input | `data/output/models/discriminator/discriminator.pt` (quindi va eseguito prima `discriminator_train`) e, con **No**, `config/discriminator_input.toml` |
| Output | nessun file: solo la finestra e il terminale |

Il significato delle uscite e l'uso da Python sono in
[discriminator_model.md](discriminator_model.md) (§9).

---

## Riepilogo

| Parte | Comando (`python -m hpfold --config config/default.toml ...`) | Richiede prima |
|---|---|---|
| GUI | `gui` | — |
| REMC | `simulate` | `gui` |
| Verifica REMC | `verify` | `simulate` |
| Dataset discriminator | `discriminator_dataset` | — |
| Addestramento discriminator | `discriminator_train` | `discriminator_dataset` |
| Giudizio di una conformazione | `discriminator_score` | `discriminator_train` |

Le altre fasi (`policy_dataset` per il dataset della policy di folding e
`train` per il suo modello) non sono trattate qui: vedi
[policy_dataset.md](policy_dataset.md).
