# Test training

Sto analizzando i file di output del training (report, history, metriche); ora passo alle predizioni sul test per calibrazione e analisi degli errori.

Ricalcolo le predizioni sul validation per scegliere la soglia senza "sbirciare" il test, e verifico la robustezza su conformazioni fuori distribuzione (random walk).

## Il training ha dato i risultati attesi?

Sì. Il modello deve dire se una conformazione è nativa, cioè se E(seq, conf) = E_min(seq), e su questo funziona bene. Il collo di bottiglia è quello previsto in §1 delle note: la stima di E_min.

I numeri di `data/output/models/discriminator/` coincidono con quelli di §7 di [discriminator_model.md](https://claude.ai/epitaxy/notes/discriminator_model.md).

### Cosa torna con le aspettative

| Controllo                        | Risultato                                                                           |
| -------------------------------- | ----------------------------------------------------------------------------------- |
| Test (6230 esempi)               | accuratezza **98,1%**, AUROC **0,9994**                                             |
| GNN contro le baseline           | GNN 98,1 > B1b 96,5 > B0b 92,0 > B0a 79,3: l'architettura aggiunge qualcosa         |
| Decoy `mid` e `far`              | 100%. Tutti gli errori sono sui nativi o sui decoy con gap = 1                      |
| Testa dell'energia               | arrotondata, giusta nel 99,95% dei casi: E la calcola davvero                       |
| Ordinamento dentro ogni sequenza | in tutte le 200 sequenze di test ogni nativo ha un punteggio più alto di ogni decoy |
| Etichette                        | `e_min` è controllata con l'enumerazione esatta, quindi i "nativi" sono veri nativi |

**Prova fuori distribuzione (fatta da me).** Ho generato 10 000 self-avoiding walk casuali sulle sequenze di test, quindi conformazioni che il modello non ha mai visto in training: i decoy del training vengono tutti dalla REMC. Il modello sbaglia 4 casi su 10 000, e riconosce come nativi 36 dei 38 veri nativi capitati per caso. Per l'uso come giudice delle conformazioni della policy questo è il risultato che conta di più.

### Cosa va tenuto d'occhio

2. **La soglia 0,5 sbilancia verso "decoy".** Gli errori sono 95 nativi rifiutati contro 22 decoy accettati. Anche le probabilità lo mostrano: quando il modello dà p tra 0,1 e 0,5, il 66–74% di quegli esempi è in realtà nativo.

   - Scegliendo la soglia sul validation (0,12) l'accuratezza sul test sale a **98,7%**, senza usare il test per sceglierla.

3. **Il calo dal validation (99,4%) al test (98,1%) viene da poche sequenze.** Tutti gli errori stanno in 23 sequenze, e 6 di queste ne contano 82 su 117.

   - Nelle sequenze peggiori la sequenza di train più vicina (1–2 residui diversi) ha una E_min più bassa. Per esempio la 1845 ha E_min = −5, mentre la sua vicina di train ha −7.
   - Il modello sembra quindi "ereditare" la E_min delle sequenze vicine. Le mutazioni singole che cambiano E_min sono il caso difficile.
   - Non c'è però una fuga di informazione dal train al test: non ci sono sequenze duplicate, e solo il 34% delle vicine ha la stessa E_min.

4. **C'è poco overfitting.** La loss BCE vale 0,011 sul train, 0,017 sul validation e 0,040 sul test; l'accuratezza è 99,6 / 99,4 / 98,1%.

   - L'AUROC di validation è quasi fermo dall'epoca 14 circa, mentre l'accuratezza di validation oscilla molto da un'epoca all'altra (tra 0,976 e 0,995).
   - Con l'AUROC già a 0,9998, l'early stopping in pratica sceglie l'epoca per rumore. Usare la BCE di validation come criterio sarebbe più informativo.

5. **Il run è stato fatto senza la correzione dei numeri denormali.** È il commit `4b081d2`, che non contiene `set_flush_denormal`.

   - Per questo le epoche passano da circa 145 s a circa 440 s: circa 3 ore in tutto, contro i circa 10 minuti attesi.
   - Riallenando con il codice attuale i numeri possono cambiare alla terza cifra decimale.

6. **I limiti d'uso.** Il modello vale solo per catene di 16 residui. Le probabilità sono tarate su un dataset con metà nativi e metà decoy, quindi per la policy vanno lette come punteggio, non come frequenza reale.



## AUROC e accuratezza

**Cosa misura.** È la probabilità che, presi a caso un nativo e un decoy, il modello dia al nativo un punteggio più alto del decoy . Il punteggio è la probabilità p che uno stato sia nativo.

- **1,0**: ogni nativo ha un punteggio più alto di ogni decoy, quindi ordinamento perfetto.
- **0,5**: il modello non distingue, è come tirare una moneta.
- **0,9994** (la GNN sul test): su 10 000 coppie nativo/decoy, solo circa 6 sono ordinate al contrario.

**Da dove viene il nome.** *Area Under the ROC curve*. Si fa variare la soglia da 1 a 0 e per ogni soglia si segna:

- quanti nativi vengono accettati (veri positivi);
- quanti decoy vengono accettati per errore (falsi positivi).

La curva che si ottiene è la ROC, e l’AUROC è l’area sotto di essa. Un modello perfetto sale subito al 100% di nativi accettati senza nessun decoy, e l’area vale 1.

### Esempio per capire l'AUROC

Consideriamo con 4 conformazioni (2 nativi N, 2 decoy D) e la probabilità di nativo stimata dal modello:

|     | N₁  | D₁  | N₂  | D₂  |
| --- | --- | --- | --- | --- |
| p   | 0,9 | 0,6 | 0,4 | 0,1 |

Consideriamo D₂, il suo p=0,6 quindi , secondo il modello, D₁ ha il 60% di probabilità di essere nativo. Il modello però sbaglia, perché D₁ è un decoy. Con la soglia 0,5 verrebbe classificato come nativo: è un falso positivo.

Per ogni soglia, il modello dichiara **nativa** ogni conformazione con p ≥ soglia, e **decoy** tutte le altre:

| Soglia | Dichiarate native | Nativi riconosciuti (veri positivi) | Decoy dichiarati nativi per errore (falsi positivi) |
| ------ | ----------------- | ----------------------------------- | --------------------------------------------------- |
| 1,0    | nessuna           | 0/2                                 | 0/2                                                 |
| 0,8    | N₁                | 1/2                                 | 0/2                                                 |
| 0,5    | N₁, D₁            | 1/2                                 | 1/2 (D₁, perché 0,6 ≥ 0,5)                          |
| 0,3    | N₁, D₁, N₂        | 2/2                                 | 1/2 (D₁)                                            |
| 0,0    | tutte             | 2/2                                 | 2/2 (D₁, D₂)                                        |

- Abbassando la soglia il modello dichiara nativo sempre di più: riconosce più nativi veri, ma sbaglia anche su più decoy.
- D₁ è dichiarato decoy (correttamente) alle soglie 1,0 e 0,8, e nativo (per errore) da 0,5 in giù.
- Ogni riga è un punto della curva ROC: in x la frazione di decoy dichiarati nativi, in y la frazione di nativi riconosciuti.
- L'area sotto la curva (AUROC) qui vale **0,75**: delle 4 coppie nativo/decoy, 3 sono ordinate bene (il nativo ha p più alta del decoy). Solo la coppia N₂ (0,4) e D₁ (0,6) è invertita.

---

#### Esempio grafico ROC

Nella ROC ogni soglia diventa un punto, con questi assi:

- **asse x, tasso di falsi positivi (FPR)**: la frazione di decoy che il modello dichiara nativi per errore;

$$
\text{FPR} = \frac{\text{decoy dichiarati nativi}}{\text{decoy totali}}
$$

- **asse y, tasso di veri positivi (TPR)**: la frazione di nativi che il modello riconosce;  

$$
\text{TPR} = \frac{\text{nativi riconosciuti}}{\text{nativi totali}}
$$

Entrambi vanno da 0 a 1. Nell'esempio i punti sono:

| Soglia | x (FPR) | y (TPR) |
| ------ | ------- | ------- |
| 1,0    | 0       | 0       |
| 0,8    | 0       | 0,5     |
| 0,5    | 0,5     | 0,5     |
| 0,3    | 0,5     | 1       |
| 0,0    | 1       | 1       |

```
TPR
 1 |        ●───────●
   |        │
0,5●────────●
   |
 0 ●───────────────── FPR
   0       0,5      1
```

- **Modello perfetto**: la curva sale subito a TPR = 1 con FPR = 0, cioè fino all'angolo in alto a sinistra, e l'area vale 1.
- **Modello casuale**: la curva segue la diagonale e l'area vale 0,5.
- **Nell'esempio**: l'area sotto la curva è 0,75.

---

### Accuratezza

È la **frazione di esempi a cui il modello dà il verdetto giusto**. Per ogni esempio ci sono tre passi:

1. il modello produce la probabilità di nativo p;
2. il verdetto è “nativo” se p ≥ 0,5, altrimenti “decoy”;
3. il verdetto si confronta con l’etichetta vera.

$$
\text{accuratezza} = \frac{\text{verdetti giusti}}{\text{esempi totali}}
$$

**Per la GNN sul test:** 6230 esempi, 117 sbagliati (95 nativi rifiutati e 22 decoy accettati), quindi (6230 − 117) / 6230 = **0,981**.

**Accuratezza per gruppo** (`native`, `near`, `mid`, `far`, `traps`): è lo stesso calcolo, fatto solo sugli esempi di quel gruppo.

- Sui **nativi** è la frazione di nativi riconosciuti come tali: 97,0%.
- Su un gruppo di **decoy** è la frazione di decoy scartati: per esempio 98,2% sui `near`.

Per questo la GNN ha un AUROC quasi perfetto ma un’accuratezza del 98,1%: ordina bene, ma la soglia 0,5 è un po’ troppo severa con i nativi (con 0,12 sale al 98,7%).

## Le baseline

Sono modelli più semplici della GNN, allenati sugli stessi dati e con gli stessi input. Servono a capire **dove sta la difficoltà** e se l’architettura della GNN aggiunge qualcosa. Il prefisso **B0** indica i modelli lineari (regressione logistica), **B1** quelli non lineari. La B1a, gradient boosting, è prevista ma non implementata.

#### B0a: solo l’energia

Una regressione logistica con un’unica feature, l’energia E della conformazione:

$$p = \sigma(a \cdot E + b)$$

In pratica sceglie una soglia di energia, del tipo “nativa se E ≤ −4”. Non può funzionare bene, perché la stessa energia è nativa per una sequenza e decoy per un’altra:

- per una sequenza con E_min = −2, la conformazione a −2 è nativa;
- per una sequenza con E_min = −7, una conformazione a −6 è un decoy.

Lo si vede nelle metriche: sulle sequenze con E_min = −3, −2, −1 ha accuratezza esattamente 0,5. Classifica tutto come decoy, quindi indovina i decoy e sbaglia tutti i nativi.

**Risultato: 79,3%.** Dice quanto si risolve guardando solo quanto è compatta la forma.

#### B0b: energia più una stima grezza di E_min

È ancora una regressione logistica, ma oltre a E riceve alcuni conteggi sulla sequenza:

- numero di H;
- H in posizione pari e H in posizione dispari;
- min(H pari, H dispari);
- H agli estremi della catena.

L’idea viene dal reticolo quadrato, che è come una scacchiera: due residui possono toccarsi solo se uno è in posizione pari e l’altro dispari. Il numero di contatti H-H possibili è quindi limitato da circa 2 · min(H pari, H dispari). Gli estremi contano a parte perché hanno un vicino libero in più.

Con queste feature il modello ha una stima, anche se rozza, di quanto in basso può arrivare l’energia della sequenza. È però un modello lineare e conosce solo dei conteggi, non l’ordine degli H lungo la catena.

**Risultato: 92,0%.** Già questo limite grezzo risolve molto, ma sui decoy `near` (gap = 1) si ferma al 75,7%.

#### B1b: MLP con le stesse informazioni della GNN

Una rete neurale semplice (2 layer nascosti) che riceve un unico vettore piatto:

- il vettore HP: 16 numeri, 1 per H e 0 per P;
- per ogni coppia di residui che può toccarsi (distanza lungo la catena dispari e almeno 3), 1 se c’è un contatto H-H. Per N = 16 sono 49 coppie.

L’informazione è la stessa della GNN, ma senza struttura a grafo: ogni coppia ha i suoi pesi, e la rete deve imparare separatamente che il contatto 0–3 e il contatto 5–8 sono “la stessa cosa”.

**Risultato: 96,5%.** Mostra che buona parte del lavoro si fa anche senza GNN.



### Come si leggono insieme

| Modello | Cosa sa                                        | Accuratezza test | Decoy `near` |
| ------- | ---------------------------------------------- | ---------------- | ------------ |
| B0a     | solo E                                         | 79,3%            | 43,9%        |
| B0b     | E + limite grezzo su E_min                     | 92,0%            | 75,7%        |
| B1b     | sequenza completa + contatti, rete piatta      | 96,5%            | 91,9%        |
| GNN     | sequenza completa + contatti, grafo + due rami | **98,1%**        | **98,2%**    |

Ogni passo aggiunge una stima migliore di **E_min**, ed è proprio lì che sta la difficoltà del problema. La GNN dimezza gli errori della B1b soprattutto sui decoy `near`, quelli che hanno un solo contatto in meno dei nativi.


