# Il decoy discriminator (fase 3b)

Questa nota descrive il modello che, data una sequenza HP e una conformazione, dice se la conformazione è **nativa** (energia minima per quella sequenza) oppure un **decoy**. Il dataset su cui si allena è descritto in [discriminator_dataset.md](discriminator_dataset.md); le idee alla base dell'architettura sono in [architettura_ML.md](architettura_ML.md).

Per allenarlo (dopo aver generato il dataset):

```bash
python -m hpfold --config config/default.toml discriminator_train
```

Serve PyTorch (`pip install -e ".[ml]"`). I file finiscono in `output.discriminator_model_dir` (di default `data/output/models/discriminator/`).

---

## 1. Cosa deve imparare il modello

L'etichetta è `label = [E(seq, conf) == E_min(seq)]`. Il compito ha quindi due parti:

1. **E(seq, conf)**, l'energia della conformazione: meno il numero di contatti H-H. È locale e si calcola esattamente se si vedono i contatti.
2. **E_min(seq)**, l'energia minima della sequenza: dipende solo dalla sequenza, ma è un'informazione globale e difficile da stimare.

Il modello ha un ramo per ciascuna delle due parti, e un classificatore che le confronta.

---

## 2. Input

Ogni modello di questa fase, baseline comprese, riceve **solo**:

- **`hp`** (N numeri): 1 per H, 0 per P;
- **`contacts`** (matrice N × N): 1 se i residui i e j sono vicini sul reticolo senza essere consecutivi nella catena. Si ricava dalle coordinate con `policy_dataset.contact_map`.

Entrambi vengono da `discriminator_dataset.model_inputs`. Le altre colonne del dataset (`tier`, `gap`, `e_min`, `kinetic_trap`, `native_overlap` e la provenienza REMC) contengono la risposta e non arrivano mai ai modelli: servono come target o per l'analisi degli errori.

**Perché la matrice dei contatti e non i turni o le coordinate.** La matrice dice solo *chi tocca chi*: non cambia se la catena viene traslata, ruotata o specchiata. Il modello è quindi invariante a queste simmetrie per costruzione (lo verifica un test), e non serve l'augmentation dei turni L ↔ R.

**Feature dei residui.** Dalla sequenza si calcolano 4 numeri per residuo (`node_features`):

| Feature   | Valore                                                      |
| --------- | ----------------------------------------------------------- |
| H         | 1 se il residuo è H                                         |
| parità    | i mod 2: sul reticolo quadrato si toccano solo pari e dispari |
| posizione | i / (N − 1)                                                 |
| estremo   | 1 per il primo e l'ultimo residuo                           |

---

## 3. Architettura

```
                 hp, contacts
                ┌──────┴───────┐
   ramo conformazione       ramo sequenza
   GNN (4 layer)            CNN 1D (4 layer)
   grafo: legami+contatti   solo la stringa HP
        │ pooling                │ pooling
      z_conf (128)             z_seq (128)
        │                        │
   testa energia → Ê        testa E_min → Ê_min
        └──────────┬─────────────┘
         classificatore: MLP su [z_conf, z_seq, z_conf ⊙ z_seq]
                   │
            logit (→ p̂ nativo), ĝ (gap)
```

**Ramo conformazione (`ConformationGNN`).** I residui sono i nodi; gli archi sono di due tipi, legami (i, i+1) e contatti. Ogni layer (`GraphLayer`):

1. **messaggio**: ogni vicino j manda `W_tipo · h_j + b_tipo`, con pesi diversi per legami e contatti;
2. **aggregazione**: i messaggi si sommano (in forma matriciale: `bonds @ … + contacts @ …`);
3. **aggiornamento**: un MLP combina il vettore del residuo con la somma dei messaggi; connessione residua e `LayerNorm`.

Con un solo layer la rete può già contare i contatti H-H di ogni residuo, cioè l'energia; i layer successivi vedono il contesto (un H isolato vicino a una zona compatta, un tratto piegato male). Dopo l'ultimo layer i 16 vettori si riassumono con **somma e massimo** (`pool`). Non si usa la media perché, con catene tutte della stessa lunghezza, sarebbe la somma divisa per N: il massimo aggiunge invece un'informazione diversa.

**Ramo sequenza (`SequenceCNN`).** Quattro convoluzioni 1D di larghezza 5 sulle stesse feature dei residui, senza la conformazione. Dopo 4 layer ogni posizione vede 4 · 4 + 1 = 17 residui, cioè tutta la catena. Stesso pooling (somma e massimo).

**Teste.** Tre MLP con un layer nascosto e dropout:

| Testa                       | Riceve                          | Predice              | Target   |
| --------------------------- | ------------------------------- | -------------------- | -------- |
| energia (`energy_head`)     | `z_conf`                        | Ê                    | `energy` |
| energia minima (`e_min_head`) | `z_seq`                       | Ê_min                | `e_min`  |
| classificatore              | `z_conf`, `z_seq`, il loro prodotto | logit di nativo, ĝ | `label`, `gap` |

Il prodotto elemento per elemento `z_conf ⊙ z_seq` rende facile al classificatore il confronto tra "quanto è buona questa forma" e "quanto potrebbe esserlo al massimo".

**Dimensioni.** Con i valori di default (`hidden = 64`) il modello ha circa 188 000 parametri.

---

## 4. Training

**Loss.**

$$
L = \text{BCE}(\text{logit}, \text{label}) + w_E\,\text{MSE}(\hat E, E) + w_{min}\,\text{MSE}(\hat E_{min}, E_{min}) + w_g\,\text{MSE}(\hat g, g)
$$

con i pesi di `loss_weights` (0,1 ciascuno). Il gradiente di ogni termine raggiunge solo le parti della rete da cui dipende la sua predizione: l'MSE dell'energia aggiorna testa energia e GNN, quello di E_min la testa E_min e la CNN, BCE e gap il classificatore e tutti e due i rami. Mettendo tutti i pesi a 0 si ottiene l'ablation "solo classificatore".

**Ottimizzazione.** Adam (`learning_rate` 1e-3, `weight_decay` 1e-4), batch da 256 esempi mescolati a ogni epoca.

**Augmentation: la catena letta al contrario.** In ogni batch metà degli esempi viene letta dall'altro estremo (`reverse_chain`: il residuo i diventa N − 1 − i, sia in `hp` sia nella matrice dei contatti). È lo stesso stato fisico, quindi energia, E_min ed etichetta non cambiano. Le rotazioni e le riflessioni non servono, perché la matrice dei contatti non le vede.

**Early stopping.** A ogni epoca si calcola l'AUROC sul validation set; si tengono i pesi dell'epoca migliore e ci si ferma dopo `patience` (10) epoche senza miglioramenti, al massimo `max_epochs` (60).

**Riproducibilità.** Pesi iniziali, ordine dei batch e augmentation dipendono da `seed`: due allenamenti con la stessa configurazione danno gli stessi numeri (lo verifica un test).

---

## 5. Baseline

Le baseline dicono **dove sta la difficoltà**: se un modello semplice va come la GNN, l'architettura non aggiunge nulla. Sono allenate sugli stessi split, con gli stessi input e le stesse metriche.

| Nome  | Modello                                     | Input                                                                                              |
| ----- | ------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `B0a` | regressione logistica                        | la sola energia E, calcolata dagli input                                                            |
| `B0b` | regressione logistica                        | E più feature della sola sequenza: numero di H, H pari, H dispari, min(H pari, H dispari), H agli estremi |
| `B1b` | MLP (2 layer nascosti)                       | il vettore HP e i contatti H-H delle coppie che possono toccarsi (j − i dispari e ≥ 3)              |

- **B0a** misura quanto si risolve guardando solo quanto è compatta la forma.
- **B0b** aggiunge una stima grezza di E_min: due residui si toccano solo se uno è pari e l'altro dispari, quindi il numero di contatti H-H è limitato da circa 2 · min(H pari, H dispari).
- **B1b** ha la stessa informazione della GNN, ma senza struttura a grafo e senza pesi condivisi tra le coppie.

Le due regressioni logistiche sono problemi convessi: si allenano a batch intero con L-BFGS, che arriva all'ottimo senza epoche (`fit_logistic`). B1b si allena come la GNN.

La baseline B1a (gradient boosting su feature ingegnerizzate) non è implementata: richiederebbe `sklearn` o LightGBM.

---

## 6. Metriche

Calcolate da `evaluate` su validation e test:

| Metrica                   | Significato                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------- |
| `accuracy`                | frazione di esempi classificati bene con soglia 0,5 sulla probabilità di nativo                    |
| `auroc`                   | probabilità che un nativo a caso abbia un punteggio più alto di un decoy a caso, su tutto lo split |
| `sequence_auroc`          | la stessa cosa dentro ogni sequenza, mediata sulle sequenze                                       |
| `accuracy_by_tier`        | accuratezza su nativi e su decoy `near`, `mid`, `far`                                              |
| `accuracy_kinetic_traps`  | accuratezza sui decoy che sono trappole cinetiche                                                  |
| `accuracy_by_e_min`       | accuratezza per energia minima della sequenza                                                      |
| `mae_energy`, `mae_e_min`, `mae_gap` | errore medio assoluto delle teste ausiliarie (solo GNN)                                |

**Attenzione a `sequence_auroc`.** Dentro una sequenza i nativi sono esattamente gli stati a energia più bassa, quindi ordinare per energia dà già `sequence_auroc = 1` (lo fa B0a). La metrica dice solo se il modello ordina bene le energie, non se sa dove sta il minimo. La difficoltà vera è la **soglia assoluta**, che si vede in `accuracy` e soprattutto nell'accuratezza sui decoy `near`.

---

## 7. Risultati

RISULTATI

---

## 8. File

| File                   | Contenuto                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------- |
| `discriminator.pt`     | iperparametri e pesi del modello principale                                                                    |
| `metrics.json`         | tutte le metriche di §6, per modello e per split, più la configurazione usata e il commit git                 |
| `report.txt`           | la tabella di confronto tra i modelli su test e validation (la stessa stampata a fine esecuzione)            |
| `history.csv`          | una riga per modello ed epoca: tempo, termini della loss di train, AUROC e accuratezza di validation          |
| `test_predictions.csv` | una riga per esempio di test: sequenza, chiave, fascia, target, probabilità di ogni modello, uscite delle teste ausiliarie della GNN |

---

## 9. Uso da Python

```python
from hpfold import discriminator_model as dm

model = dm.load_model("data/output/models/discriminator")
p = dm.score(model, ["HHPHHPHHPPHPPHHP"], coordinates)   # coordinates: (S, N, 2)
```

`score` restituisce la probabilità di nativo di ogni conformazione: è l'uso del modello come giudice, per esempio delle conformazioni proposte dalla policy. Le coordinate possono essere in qualsiasi posizione e orientamento.

---

## 10. Parametri (`[discriminator_model]`)

| Parametro         | Default              | Significato                                                              |
| ----------------- | -------------------- | ------------------------------------------------------------------------ |
| `seed`            | 0                    | pesi iniziali, ordine dei batch, augmentation                            |
| `hidden`          | 64                   | dimensione dei vettori dei residui e dei layer nascosti                  |
| `gnn_layers`      | 4                    | layer di message passing del ramo conformazione                          |
| `cnn_layers`      | 4                    | layer convoluzionali del ramo sequenza                                   |
| `cnn_kernel`      | 5                    | residui visti da un filtro (dispari)                                     |
| `dropout`         | 0,1                  | nelle teste MLP                                                          |
| `loss_weights`    | 0,1 / 0,1 / 0,1      | pesi degli MSE di energia, E_min e gap; 0 spegne una testa               |
| `learning_rate`   | 1e-3                 | Adam                                                                     |
| `weight_decay`    | 1e-4                 | Adam                                                                     |
| `batch_size`      | 256                  |                                                                          |
| `max_epochs`      | 60                   |                                                                          |
| `patience`        | 10                   | epoche senza un AUROC di validation migliore prima di fermarsi          |
| `augment_reverse` | true                 | metà di ogni batch letta dall'altro estremo della catena                 |
| `train_sequences` | 0                    | sequenze di train usate, scelte a caso (0 = tutte): per le learning curve |
| `baselines`       | `["B0a", "B0b", "B1b"]` | baseline allenate accanto al modello (`[]` = nessuna)                 |
| `threads`         | 0                    | thread CPU di torch (0 = default di torch)                               |

---

## 11. Dove sta il codice

| File                                                                   | Contenuto                                                                    |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [discriminator_model.py](../python/hpfold/discriminator_model.py)      | input, modello, baseline, training, metriche, file                            |
| [discriminator_dataset.py](../python/hpfold/discriminator_dataset.py)  | `load_split`, `model_inputs`: lettura del dataset                            |
| [policy_dataset.py](../python/hpfold/policy_dataset.py)                | `contact_map`                                                                |
| [test_discriminator_model.py](../tests/test_discriminator_model.py)    | i test (§12)                                                                 |

Le funzioni principali di `discriminator_model.py`:

- `load_examples`: uno split come tensori (input, target e informazioni per l'analisi);
- `node_features`, `reverse_chain`, `hh_energy`, `sequence_features`: input e feature;
- `Discriminator` (con `ConformationGNN`, `SequenceCNN`, `GraphLayer`), `Logistic`, `PairMLP`: i modelli;
- `fit`, `fit_logistic`, `predict`: training e predizione;
- `auroc`, `evaluate`, `report`: metriche;
- `run`, `load_model`, `score`: la fase completa e l'uso del modello allenato.

---

## 12. Cosa verificano i test

`tests/test_discriminator_model.py` (saltato se torch non è installato), su un dataset piccolo (8 sequenze di 10 residui) e un modello piccolo:

- le feature dei residui e della sequenza su casi noti;
- l'energia calcolata dagli input coincide con quella del dataset;
- ai modelli arrivano solo `hp` e la matrice dei contatti;
- la probabilità prevista non cambia ruotando o specchiando le coordinate;
- `reverse_chain` coincide con la matrice dei contatti della catena letta al contrario, e l'energia non cambia;
- `auroc` coincide con il conteggio su tutte le coppie nativo/decoy, anche con molti pareggi;
- la fase scrive tutti i file e le metriche di tutti i modelli;
- il modello riletto da `discriminator.pt` ridà le probabilità di `test_predictions.csv`, anche passando da sequenze e coordinate con `score`;
- due allenamenti con la stessa configurazione danno le stesse metriche;
- `train_sequences` limita le sequenze di train;
- la fase `discriminator_train` esiste nella riga di comando.
