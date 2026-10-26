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

Dataset di default: 2000 sequenze di 16 residui (1600 train, 200 validation, 200 test), 50 354 esempi di train e 6230 di test. Configurazione di default.

### 7.1 Confronto sul test

```
modello  accuratezza  AUROC   nativi  near   mid    far    trappole
GNN      0,981        0,9994  0,970   0,982  1,000  1,000  0,980
B1b      0,965        0,9950  0,963   0,919  1,000  1,000  0,935
B0b      0,920        0,9750  0,936   0,757  1,000  1,000  0,793
B0a      0,793        0,8710  0,835   0,439  0,911  1,000  0,577
```

(le colonne da "nativi" a "trappole" sono l'accuratezza dentro ciascun gruppo)

- **B0a → B0b** (da 79% a 92%): sapere qualcosa della sequenza conta molto. La sola energia non basta, perché la stessa energia è nativa per una sequenza e un decoy per un'altra; già un limite grezzo su E_min dalle parità degli H risolve gran parte dei casi.
- **B0b → B1b → GNN** sulla fascia `near` (76% → 92% → 98%): i decoy difficili richiedono una stima di E_min molto più fine, che i modelli appresi ricavano dalla sequenza completa.
- **B1b → GNN** (96,5% → 98,1%, metà degli errori): la struttura a grafo e i due rami separati aiutano, anche se un MLP con le stesse informazioni va già bene.
- `mid` e `far` sono risolti da tutti i modelli tranne B0a: la difficoltà è tutta nei nativi e nei `near`.

Sul validation la GNN arriva al 99,4% di accuratezza, ma quel numero è un po' ottimista: il validation set sceglie l'epoca migliore (early stopping). Il numero onesto è quello del test.

### 7.2 Training

La GNN si è fermata all'epoca 34, con i pesi migliori all'epoca 24 (AUROC di validation 0,9998). B1b si è fermata all'epoca 37 (migliore la 27).

**Tempi e numeri denormali.** I primi run erano lentissimi: dopo una decina di epoche il tempo per epoca della GNN saliva di 10 volte (con 1000 sequenze da circa 11 a circa 117 secondi), anche con un solo training sulla macchina. La causa sono i numeri *denormali*, float più piccoli di circa 1e-38, che la CPU elabora molto più lentamente. Ora `run` chiama `torch.set_flush_denormal(True)`, che li tratta come 0. Con 1000 sequenze ogni epoca resta intorno agli 11 secondi e il training passa da 46 a 7 minuti, con risultati praticamente uguali (AUROC sul test 0,9990 invece di 0,9991). Con 1600 sequenze ci si aspetta circa 18 secondi per epoca, cioè una decina di minuti in tutto.

I numeri di questa sezione vengono dai run fatti prima della correzione. Riallenando con il codice attuale possono cambiare alla terza cifra decimale, perché i valori denormali azzerati modificano di pochissimo i calcoli.

Due training lanciati insieme si contendono i core e rallentano entrambi: conviene lanciarne uno alla volta, oppure limitare `threads`.

### 7.3 Dove sbaglia la GNN

- **Le energie le sa calcolare.** La testa dell'energia, arrotondata all'intero, coincide con l'energia vera nel 99,95% degli esempi di test (errore medio 0,13).
- **Il limite è E_min.** La testa dell'energia minima ha un errore medio di 0,27: arrotondata, è giusta per 156 sequenze di test su 200.
- **Gli errori sono pochi e concentrati.** Sono 117 su 6230: 95 nativi classificati come decoy e 22 decoy `near` classificati come nativi. Riguardano 23 sequenze su 200, e le 6 peggiori contano 82 errori.
- **Il tipo di errore è sempre lo stesso.** Nelle sequenze peggiori la rete stima un'energia minima più bassa di quella vera (per esempio −5,4 invece di −5). Crede che si possa fare un contatto in più, e quindi scarta come decoy i nativi veri.

È il collo di bottiglia previsto in §1. Per migliorare servono soprattutto più sequenze, o informazioni migliori sull'energia minima, più che una rete più grande. La learning curve (`train_sequences` = 100, 250, 500, 1000, 1600) dice se aggiungere sequenze serve ancora.

### 7.4 Learning curve

Stessa configurazione, ma usando solo una parte delle sequenze di train, scelte a caso (`train_sequences`). Validation e test restano sempre gli stessi (200 + 200 sequenze). Il punto da 1600 è il run completo di §7.1. Risultati sul test:

```
sequenze   GNN                                        B1b      B0b
di train   accuratezza  AUROC   nativi  near   MAE E_min  accuratezza  accuratezza
   100     0,910        0,9811  0,845   0,937  0,63       0,877        0,911
   250     0,951        0,9925  0,940   0,907  0,43       0,927        0,910
   500     0,943        0,9947  0,901   0,963  0,37       0,939        0,923
  1000     0,989        0,9991  0,996   0,956  0,33       0,967        0,917
  1600     0,981        0,9994  0,970   0,982  0,27       0,965        0,920
```

(B0a vale sempre 0,793: con una sola feature, l'energia intera, la soglia scelta non cambia con il numero di sequenze.)

**Come leggerla.**

- **L'AUROC della GNN cresce sempre, ma si sta saturando.** Passa da 0,981 a 0,999 tra 100 e 1000 sequenze; da 1000 a 1600 guadagna solo 0,0003.
- **L'accuratezza oscilla** (per esempio 0,989 a 1000 e 0,981 a 1600). Dipende dalla soglia fissa a 0,5: a 1000 sequenze il modello è più "generoso" con i nativi (99,6% giusti, `near` 95,6%), a 1600 più severo (97,0% e 98,2%). Inoltre gli errori si concentrano in poche sequenze di test, quindi una sola sequenza sposta l'accuratezza di qualche decimo di punto. Per confrontare i punti della curva conviene guardare l'AUROC.
- **L'errore su E_min scende ancora** (0,63 → 0,27) e non ha smesso di migliorare: è la parte del modello che beneficia di più sequenze, ed è il collo di bottiglia di §7.3.
- **Le baseline si fermano prima.** B0b è ferma intorno al 91–92% già da 100 sequenze: il limite è il modello, non i dati. B1b cresce fino a 1000 sequenze (96–97%) e poi si ferma.
- **Con poche sequenze la GNN vale quanto B0b** (91% con 100 sequenze, ma con un AUROC molto più alto). La differenza tra le due cresce con i dati.

**Conclusione.** Con 1600 sequenze la GNN è vicina al limite di questo dataset per l'ordinamento (AUROC), ma la stima di E_min migliora ancora. Portare il dataset a circa 5000 sequenze (circa 75 minuti di generazione) dovrebbe ridurre gli errori residui sui nativi, con guadagni piccoli: pochi decimi di punto di accuratezza.

**Attenzione: le sequenze di test sono vicine a quelle di train.** Ognuna differisce da una sequenza di train (o dalla sua inversa) per 1 o 2 residui (vedi la nota sul dataset). Il 98% vale quindi per sequenze "nuove ma simili": su sequenze lontane da quelle viste, per esempio più lunghe, il modello va rimisurato.

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

### 9.1 Uso da riga di comando

```bash
python -m hpfold --config config/default.toml discriminator_score
```

La fase carica il modello salvato da `discriminator_train` e apre una finestra che chiede se disegnare la conformazione da giudicare, come la fase `gui` per la configurazione iniziale del REMC:

- **Yes**: si apre l'editor della fase 0 (catena di `discriminator_dataset.length` residui, poi i tipi H/P). Il pulsante **Evaluate** mostra il giudizio; la finestra resta aperta, così si possono provare più conformazioni una dopo l'altra.
- **No**: si usa la conformazione del file `discriminator_model.input_conformation`, di default [config/discriminator_input.toml](../config/discriminator_input.toml), un file TOML a parte con `sequence` e `coordinates`.
- **Cancel**: esce senza fare nulla.

Il risultato compare in una finestra ed è stampato anche sul terminale: verdetto (nativo se `prob` ≥ 0,5), probabilità di nativo, logit, energia prevista accanto a quella esatta (meno i contatti H-H), energia minima prevista della sequenza e gap previsto. La conformazione è controllata prima di arrivare al modello: deve avere la lunghezza su cui il modello è stato allenato ed essere un self-avoiding walk.

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
| `input_conformation` | `config/discriminator_input.toml` | conformazione giudicata da `discriminator_score` se non la si disegna (§9.1) |

---

## 11. Dove sta il codice

| File                                                                   | Contenuto                                                                    |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| [discriminator_model.py](../python/hpfold/discriminator_model.py)      | input, modello, baseline, training, metriche, file                            |
| [discriminator_dataset.py](../python/hpfold/discriminator_dataset.py)  | `load_split`, `model_inputs`: lettura del dataset                            |
| [policy_dataset.py](../python/hpfold/policy_dataset.py)                | `contact_map`                                                                |
| [discriminator_score.py](../python/hpfold/discriminator_score.py)      | fase `discriminator_score`: file di input, controlli, finestra del giudizio (§9.1) |
| [test_discriminator_model.py](../tests/test_discriminator_model.py)    | i test (§12)                                                                 |
| [test_discriminator_score.py](../tests/test_discriminator_score.py)    | i test della fase `discriminator_score`                                      |

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
