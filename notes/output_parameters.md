# Output Parameters

Il seguente è un esempio di output che viene mostrato a terminale dopo aver inserito una configurazione in input al modello. Lo stesso testo compare nel riquadro **Model prediction**, in basso a destra nella finestra dell'animazione e, se la configurazione è stata disegnata, anche nell'editor.
La configurazione di input è una confogurazione lineare (per questo `Probability of native: 0.0000`)


```bash
Sequence: PPHHPHPPPPHHPPPP
Verdict: decoy
Probability of native: 0.0000
Logit: -41.036
Energy (predicted): -0.09   exact: 0
Lowest energy of the sequence (predicted): -2.60
Gap from the lowest energy (predicted): 2.82
```

## Verdict, Probability of native e Logit

Il verdetto si ottiene in tre passi: il modello calcola un numero (il **logit**), lo trasforma in una **probabilità** e poi applica una **soglia**.

```
conformazione + sequenza  →  rete neurale  →  logit  →  sigmoide  →  probabilità  →  soglia 0.5  →  verdetto
                                              -41.036               0.0000                        decoy
```

#### Passo 1 (logit): la rete produce il logit

Il modello riceve due input:

- la **sequenza HP**, cioè per ogni residuo 1 se è H e 0 se è P;
- la **mappa dei contatti** della conformazione, cioè quali residui sono vicini sul reticolo.

I due rami elaborano queste informazioni:

- il ramo conformazione (GNN) riassume la struttura nel vettore `z_conf`, che contiene in sostanza "quanti e quali contatti H-H ci sono";
- il ramo sequenza (CNN) riassume la sequenza nel vettore `z_seq`, che contiene in sostanza "quanto in basso potrebbe arrivare l'energia di questa sequenza".

Il classificatore (un MLP) riceve `z_conf`, `z_seq` e il loro prodotto `z_conf * z_seq`, e restituisce un solo numero reale: il **logit**. Il logit può valere qualsiasi cifra, da −∞ a +∞:

- molto positivo: "sembra nativa";
- molto negativo: "sembra un decoy";
- vicino a 0: "non so".

#### Passo 2 (probability of native): la sigmoide trasforma il logit in probabilità

Il logit non è una probabilità, perché non sta tra 0 e 1. Per ottenerla si usa la **funzione sigmoide**:

$$
p = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}
$$

La sigmoide comprime qualsiasi numero nell'intervallo (0, 1):

| logit   | p (probabilità di nativa) |
| ------- | ------------------------- |
| +5      | 0.993                     |
| +2      | 0.881                     |
| 0       | **0.500**                 |
| −2      | 0.119                     |
| −5      | 0.007                     |
| **−41** | ≈ 1.6 × 10⁻¹⁸             |

Nel tuo caso:

$$
p = \frac{1}{1 + e^{41.036}} \approx \frac{1}{6.6 \times 10^{17}} \approx 1.5 \times 10^{-18}
$$

Stampato con 4 decimali diventa `0.0000`.

#### Passo 3 (verdict): la soglia decide il verdetto

Nel codice ([discriminator_score.py:76](https://claude.ai/epitaxy/python/hpfold/discriminator_score.py:76)):

```python
verdict = "native" if result["prob"] >= 0.5 else "decoy"
```

Siccome σ(0) = 0.5, **p ≥ 0.5 equivale a logit ≥ 0**. In pratica il verdetto dipende solo dal segno del logit, e il tuo è negativo, quindi `decoy`.

### Perché quel numero è proprio una "probabilità di essere nativa"

Durante l'addestramento ogni esempio aveva un'etichetta:

- **1** se la conformazione era nativa (`energy == e_min`);
- **0** se era un decoy.

La rete è stata addestrata con la **binary cross-entropy** (`binary_cross_entropy_with_logits`, in [discriminator_model.py:296](https://claude.ai/epitaxy/python/hpfold/discriminator_model.py:296)). Questa loss penalizza le risposte sbagliate e sicure:

- se un decoy riceve p ≈ 1, la penalità è molto grande;
- se riceve p ≈ 0, è quasi nulla.

Minimizzando questa loss su migliaia di esempi, la rete impara a produrre logit tali che σ(logit) approssima la frequenza con cui conformazioni "di quel tipo" sono native. Per questo p si può leggere come una probabilità.



## Energy, Lowest energy,  Gap

Le tre grandezze rispondono a tre domande diverse:

| Grandezza         | Domanda                                                           | Dipende da                         |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------- |
| **Energy**        | Quanto è buona *questa* conformazione?                            | sequenza + conformazione disegnata |
| **Lowest energy** | Qual è la conformazione migliore *possibile* per questa sequenza? | solo la sequenza                   |
| **Gap**           | Quanto dista questa conformazione dalla migliore?                 | entrambe                           |

### Energy: il punteggio della tua conformazione

Nel modello HP l'energia è

$$
E = -(\text{numero di contatti H-H})
$$

Un contatto H-H si ha quando due residui H sono **vicini sul reticolo ma non consecutivi nella catena**. Ogni contatto vale −1, e più l'energia è bassa più la conformazione è stabile.

Vengono mandate in output due tipi di energie

- *exact:* Energia esatta della conformaizone.
- *predicted:* la stima del modello

### Lowest energy: il miglior punteggio raggiungibile

 Stima del ripiegamento migliore di della sequenza in ingresso abbia circa 2–3 contatti H-H.

### Gap

È la differenza tra l'energia predetta dal modello e l'energia minima della sequenza sempre predetta dal modello. 
Il modello non fa una semplice sottrazzione ma la predeice direttamente nel modello

**Si potrebbe fare anche una semplice sottrazione**

POSSIBILITÀ: NON MOSTRARE IL GAP



## Dopo la previsione: percorso verso il nativo e animazione

Subito dopo il giudizio, il programma calcola come portare la conformazione inserita fino allo stato nativo e lo mostra in un'animazione. **Questa parte non usa il modello ML**: usa l'**oracolo** esatto, lo stesso che etichetta il policy dataset ([folding_path.py](../python/hpfold/folding_path.py)).

### Calcolo delle mosse corrette

1. **Grafo delle mosse.** Si costruisce il grafo di tutte le conformazioni possibili di 16 residui: un nodo per ogni conformazione e una freccia per ogni mossa. Il grafo non dipende dalla sequenza, quindi si costruisce una volta sola (circa 5 s, la prima volta).
2. **Distanze.** Gli stati nativi sono tutte le conformazioni con l'energia minima della sequenza. Una BFS all'indietro, partendo da loro, dà a ogni conformazione la sua **distanza**: il numero minimo di mosse per arrivare a un nativo.
3. **Discesa.** Dalla conformazione inserita, a distanza *d*, si prova ogni mossa e si tiene una di quelle che portano a distanza *d − 1* (le **mosse ottime**). Se ce n'è più di una si sceglie quella che abbassa di più l'energia. Si ripete fino a *d* = 0.

L'ultima conformazione è **lo stato nativo associato**: il nativo più vicino in numero di mosse. Le mosse sono applicate alla catena così come è stata disegnata, quindi i residui che una mossa non tocca restano fermi.

Il percorso viene stampato sul terminale. Esempio per la stessa sequenza, partendo dalla catena dritta:

```bash
Lowest energy of the sequence (exact): -3   native states: 514
Moves to the nearest native state: 4
  start: energy 0
  1. PULL_PREV_L at residue 7: energy 0, 3 moves left
  2. PULL_PREV_L at residue 8: energy 0, 2 moves left
  3. PULL_NEXT_R at residue 9: energy -1, 1 moves left
  4. PULL_NEXT_R at residue 7: energy -3, 0 moves left
Native state: [[4, 0], [5, 0], ...]
```

- *Lowest energy (exact)* è l'energia minima vera, da confrontare con la *Lowest energy (predicted)* del modello (−2,60).
- *native states* conta le conformazioni diverse con quell'energia.
- L'energia non scende per forza a ogni mossa (mosse 1 e 2): il criterio è il **numero di mosse**, non l'energia.

### Animazione

Si apre una finestra che parte dalla conformazione inserita. Mentre il percorso viene calcolato (circa 6 secondi la prima volta, poi subito), al centro compare un riquadro con scritto "Computing the path to the native state...", che sparisce quando parte l'animazione. Ogni mossa fa scorrere i residui verso la posizione successiva, poi la catena si ferma un attimo (i tempi sono `animation_move_ms` e `animation_pause_ms` in `[gui]`).

- La barra in basso indica la mossa, il residuo, l'energia e le mosse rimaste.
- Le linee tratteggiate rosse sono i contatti H-H.
- **Pause/Play** (o spazio) ferma l'animazione e la fa riprendere da dove era.
- Ad animazione ferma, **Previous move** e **Next move** (o ← e →) fanno scorrere la catena di una mossa indietro o avanti.
- **Restart** ricomincia dall'inizio.

Se la conformazione è già nativa non ci sono mosse e la finestra lo dice.
