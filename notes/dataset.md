# Il dataset (fase 2)

Questa nota raccoglie tutto quello che riguarda il dataset: a cosa serve, come è fatto, come viene generato, perché è stato costruito così e come si usa. Le sezioni si possono leggere in ordine; la §2 spiega le parole che ritornano in tutto il resto.

Per generarlo:

```bash
python -m hpfold --config config/default.toml dataset
```

I file finiscono nella cartella `output.dataset_dir` (di default `data/output/dataset/`).

---

## 1. A cosa serve

Nella fase 3 un modello di machine learning dovrà imparare a **ripiegare la proteina un passo alla volta**: parte da una conformazione srotolata e, mossa dopo mossa, la porta fino allo stato nativo (la conformazione a energia minima).

A ogni passo il modello riceve:

1. **la sequenza HP**, che non cambia durante il ripiegamento. Esempio: `HPHHPH` diventa `[1, 0, 1, 1, 0, 1]`;
2. **la conformazione attuale**, cioè la forma della catena in quel momento;
3. facoltativamente, **l'energia** della conformazione attuale.

e deve rispondere con **una mossa**, per esempio "sposta il monomero 3 con una pull move verso sinistra".

Il dataset fornisce gli esempi da cui imparare. Ogni esempio (un **record**) è una coppia (sequenza, conformazione), insieme a:

- l'elenco delle mosse **possibili** in quella conformazione;
- per ogni mossa possibile, **quanto si avvicina allo stato nativo**;
- per ogni mossa possibile, **di quanto cambia l'energia**.

Da questi dati si ricava la risposta giusta: le mosse migliori sono quelle che avvicinano di più allo stato nativo.

---

## 2. Parole usate in questa nota

| Parola                        | Significato                                                                                                                                                                          |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Conformazione** / **stato** | La forma della catena sul reticolo: dove si trova ogni residuo.                                                                                                                      |
| **Mossa** / **azione**        | Uno spostamento di uno o più residui che porta da una conformazione valida a un'altra. Le mosse sono quelle già implementate in fase 1: VSHD (end, corner, crankshaft) e pull moves. |
| **Stato nativo**              | Una conformazione con l'energia più bassa possibile per quella sequenza. Spesso ce n'è più di una (vedi §7.3).                                                                       |
| **Distanza** di uno stato     | Il numero minimo di mosse per arrivare da quello stato a uno stato nativo. Uno stato nativo ha distanza 0.                                                                           |
| **Mossa ottima**              | Una mossa che porta in uno stato con distanza più piccola di 1. Da uno stato possono esserci più mosse ottime.                                                                       |
| **Oracolo**                   | Il programma che calcola le distanze e quindi le mosse ottime. È "l'insegnante" del dataset (vedi §7).                                                                               |
| **Grafo delle mosse**         | Un grafo in cui ogni nodo è una conformazione e c'è una freccia da A a B se una mossa porta da A a B.                                                                                |
| **BFS** (ricerca in ampiezza) | L'algoritmo che trova i cammini più corti in un grafo: visita prima tutti i nodi a distanza 1, poi quelli a distanza 2, e così via.                                                  |
| **Decoy**                     | Una conformazione non nativa usata come punto di partenza. Qui i decoy sono presi dalla REMC.                                                                                        |
| **Walker**                    | Una delle conformazioni che la REMC fa evolvere. Con gli scambi tra repliche un walker cambia temperatura, ma la sua forma cambia solo con le mosse (vedi §8).                       |
| **Slot**                      | Una posizione fissa nel vettore delle azioni: ogni slot corrisponde sempre allo stesso tipo di mossa sullo stesso residuo (vedi §6).                                                 |
| **Maschera**                  | Per ogni slot, un sì/no: la mossa di quello slot è possibile in questa conformazione?                                                                                                |

---

## 3. In generale: cosa salva la REMC durante la simulazione

Per ogni sequenza del dataset viene lanciata una REMC, partendo dalla catena dritta. Non è l'esecuzione della fase 1: è una nuova esecuzione dello stesso algoritmo, più corta e con temperature più alte (§9.2). Mentre gira, la REMC registra due cose.

1. **Il percorso di ogni catena simulata (traiettoria del walker).** Dopo ogni mossa accettata si annota la forma raggiunta. Ne esce il "filmato" del ripiegamento: dalla catena dritta, passo dopo passo, fino allo stato nativo, con tutti i giri a vuoto e le mosse in salita tipici del Monte Carlo. Il filmato continua anche dopo lo stato nativo; viene tagliato al primo stato nativo durante la pulizia (§8).
2. **Delle fotografie a intervalli regolari (snapshot).** Ogni pochi cicli si salva la forma di ogni replica, a ciascuna temperatura. Le repliche fredde danno forme compatte, quasi native; quelle calde danno forme srotolate.

La differenza tra i due salvataggi sta in **cosa si segue** e **ogni quanto si salva**.

### 3.1 Cosa si segue

Nella REMC ci sono 5 **temperature**, fisse, e 5 **conformazioni** che si muovono. Ogni tanto due conformazioni si scambiano la temperatura.

- **La traiettoria segue una conformazione** (il walker), qualunque temperatura abbia in quel momento. È come seguire un corridore che cambia corsia.
- **Lo snapshot guarda una temperatura**, qualunque conformazione ci sia in quel momento. È come fotografare sempre la stessa corsia, chiunque ci stia correndo.

*Esempio.* Il walker 2 parte alla temperatura più fredda; a metà simulazione viene scambiato e passa a una più calda. La sua traiettoria continua senza interruzioni, perché lo scambio non ne cambia la forma. Le foto della temperatura fredda, invece, da quel momento mostrano un'altra catena.

### 3.2 Ogni quanto si salva

- **Traiettoria: dopo ogni mossa accettata.** Due stati consecutivi distano esattamente una mossa, quindi il percorso è continuo.
- **Snapshot: ogni `save_every` cicli.** Con i valori di default è ogni 2 cicli da 100 passi, cioè ogni 200 passi tentati. Tra due foto la catena ha fatto molte mosse: le foto sono stati sparsi, non un percorso.

### 3.3 Cosa contiene ciascun salvataggio

|                      | Traiettoria                                           | Snapshot                                   |
| -------------------- | ----------------------------------------------------- | ------------------------------------------ |
| Informazioni salvate | solo la forma (chiave canonica, §5.2)                 | forma, energia, temperatura, ciclo, walker |
| Quante per sequenza  | fino a 50 000 stati per walker (`max_recorded_steps`) | 100 foto per temperatura, 500 in tutto     |
| Stati consecutivi    | a una mossa di distanza                               | a molte mosse di distanza                  |

### 3.4 A cosa serve ciascuno

- **Le traiettorie** servono perché sono continue. Si possono ripulire (togliere i giri a vuoto, trovare le scorciatoie) e misurare quanto si accorciano (§8). Gli stati ripuliti entrano nel dataset come percorsi `remc_pruned`. Per catene più lunghe di 16 residui, i loro stati formano anche il grafo che usa l'oracolo (§7.5).
- **Gli snapshot** servono perché sono etichettati con la temperatura. Da essi si scelgono i **punti di partenza** (decoy), un certo numero per ogni temperatura. Così si hanno sia forme srotolate (temperature calde) sia forme quasi native (temperature fredde), e da ognuna si calcola il percorso ottimo verso lo stato nativo (§9.2).

**In sintesi:** la traiettoria è il **filmato di una catena**, gli snapshot sono **foto periodiche di ogni temperatura**. La prima serve per i percorsi, i secondi per scegliere da dove partire.

Nessuno dei due salvataggi fornisce le "mosse giuste": le mosse fatte dalla REMC sono casuali e spesso inutili. La REMC fornisce solo **stati**; le etichette di ogni stato le calcola l'oracolo (§7).

---

## 4. Punto di partenza: cosa c'era nel codice

Prima di progettare il dataset ho controllato il codice di fase 0 e 1:

- **Il reticolo è 2D e quadrato**, di dimensione finita (30 × 30 in `[lattice]`). Ha dei bordi veri: un residuo non può uscire dalla griglia.
- **Le conformazioni sono salvate come coordinate assolute** `(x, y)`, una per residuo, in ordine di catena (è il formato del JSON di fase 0 e della classe `hp::Conformation`).
- **Una mossa è una lista di residui spostati**, ognuno con la sua nuova posizione. `vshd_moves(c, k)` e `pull_moves(c, k)` restituiscono le mosse possibili sul residuo `k`; il loro numero cambia da conformazione a conformazione e non c'era un modo per numerarle.
- **La REMC non salvava nulla** oltre alla conformazione migliore: niente traiettorie e niente stati intermedi.

Per costruire il dataset sono quindi servite quattro cose nuove: un modo canonico per descrivere una conformazione (§5), una numerazione fissa delle mosse (§6), un oracolo che calcoli le mosse ottime (§7) e la registrazione delle traiettorie REMC (§8).

---

## 5. Come si descrive una conformazione

### 5.1 Il problema di partenza: perché non usare le coordinate $(x, y)$

Se si descrivesse la catena usando le coordinate cartesiane di ciascun residuo:

- **Molteplicità di descrizioni per la stessa forma:** Traslando una catena sulla griglia o ruotandola nei 4 orientamenti possibili, le coordinate cambiano continuamente anche se la forma fisica della proteina è identica. 

- **Ambiguità di altre rappresentazioni:** La matrice di contatto non permette sempre di ricostruire la forma (ad esempio, tutte le conformazioni aperte senza contatti avrebbero una matrice vuota identica), mentre la griglia occupa molta memoria.

Per risolvere questi problemi, il dataset memorizza solo i **turni** (le svolte relative) in **forma canonica**.

---

Come mai avere molteplici descrizioni per la stessa forma geometrica di una proteina rappresenterebbe un problema critico per il dataset e per l'addestramento del modello?

**Incapacità di identificare e deduplicare gli stati:** Per costruire il grafo delle mosse, calcolare la distanza minima dallo stato nativo tramite la ricerca in ampiezza (BFS) e pulire le traiettorie dai cicli inutili, il sistema deve poter verificare istantaneamente se due conformazioni rappresentano lo stesso stato fisico. Se la stessa forma generasse descrizioni diverse (ad esempio a causa di traslazioni sulla griglia, rotazioni o riflessioni speculari), il programma le tratterebbe come stati distinti, fallendo nel riconoscere i duplicati e alterando il calcolo dei percorsi ottimi.

Ma se non vogliamo avere duplicati (rotazioni o simili per la stessa configurazione) come mai successivamente applicamo l'**augmentation del dataset**?
L'augmentation non contraddice la necessità di evitare duplicati, poiché viene applicata esclusivamente **al volo durante l'addestramento (in memoria)** e non durante la costruzione o il salvataggio del dataset su disco.
Sul disco, ogni conformazione viene memorizzata una sola volta in forma canonica per garantire l'efficienza degli algoritmi di calcolo del grafo, la pulizia delle traiettorie e la deduplicazione.

Tuttavia, poiché la forma canonica impone per convenzione che il primo turno non dritto sia sempre una svolta a sinistra (LEFT), un modello allenato direttamente su questi dati grezzi svilupperebbe un forte pregiudizio (bias) e non saprebbe come comportarsi davanti a catene che iniziano a piegarsi verso destra.

L'augmentation risolve questo problema al momento del caricamento tramite i seguenti meccanismi:

- **Applicazione dinamica sui batch:** L'operazione viene eseguita casualmente (ad esempio su metà dei record a ogni epoca) durante la generazione dei batch, senza alterare i file salvati o creare duplicati permanenti sul disco.

- **Trasformazione coerente di stato e mosse:** Quando una conformazione viene specchiata, si invertono le svolte nei turni (scambiando LEFT e RIGHT) e contestualmente si applica una permutazione agli slot delle mosse disponibili (ad esempio scambiando lo slot della mossa `PULL_NEXT_L` con quello di `PULL_NEXT_R`).

- **Invarianza delle etichette fisiche:** Poiché la riflessione speculare non modifica le proprietà fisiche della catena, l'energia dello stato e la distanza minima che lo separa dallo stato nativo rimangono esattamente le stesse.

In questo modo, la memorizzazione univoca su disco protegge la correttezza del grafo e del calcolo dell'oracolo, mentre l'augmentation in memoria forza la rete neurale a imparare l'invarianza speculare senza dover raddoppiare lo spazio di archiviazione.

---

### 5.2. I Turni (Rappresentazione Relativa)

Anziché dire *dove* si trova ogni residuo sulla griglia, la catena viene descritta percorrendola dal primo residuo (0) all'ultimo. A ogni residuo interno, la catena può fare una di tre mosse rispetto alla direzione da cui proviene:

- **FORWARD (codice 0):** Prosegue dritta.

- **LEFT (codice 1):** Gira a sinistra.

- **RIGHT (codice 2):** Gira a destra.

Per una catena di $N$ residui, il primo e l'ultimo residuo non svolgono; servono quindi esattamente **$N - 2$ turni** per definire l'intera conformazione.

> **Proprietà:** Questa sequenza di svolte non cambia se la proteina viene spostata o ruotata sulla griglia. Due conformazioni ruotate o traslate hanno la stessa identica sequenza di turni.

**Esempio.** La catena con coordinate `(0,0) (1,0) (2,0) (2,1) (1,1)` va a destra, ancora a destra, poi in giù, poi a sinistra. I turni sono `[FORWARD, LEFT, LEFT]`.

I turni non cambiano se la catena viene traslata o ruotata: due catene che differiscono solo per la posizione o per l'orientamento sulla griglia hanno gli stessi turni e, per il modello, sono giustamente lo stesso stato.

### 5.2 La forma canonica e la chiave

Affrontaimo il seguente problema: Come rappresentare una conformazione della proteina in modo univoco, compatto e facilmente confrontabile, eliminando le ridondanze dovute alla riflessione speculare?

Rappresentare una catena tramite le sole svolte (turni: FORWARD = 0, LEFT = 1, RIGHT = 2) permette già di ignorare la sua posizione nello spazio e la sua rotazione. Tuttavia, i turni non risolvono l'effetto specchio: se si riflette una conformazione come in uno specchio, la catena mantiene la stessa identica energia e la stessa geometria reale, ma la sua sequenza di turni cambia poiché ogni svolta a sinistra (LEFT) diventa una svolta a destra (RIGHT) e viceversa. Per evitare che lo stesso stato fisico venga salvato due volte con due rappresentazioni diverse, si applica la convenzione della **forma canonica**:

- Tra una conformazione e la sua immagine speculare, si sceglie come rappresentazione ufficiale quella in cui il primo turno "non dritto" (ovvero il primo turno diverso da FORWARD) è una svolta a sinistra (**LEFT**).

- Se il primo turno angolato della catena è una svolta a destra (**RIGHT**), la forma non è canonica: viene quindi riflessa invertendo tutti i LEFT con i RIGHT.

- L'unica eccezione è la catena completamente dritta (composta solo da FORWARD), che non possiede un'immagine speculare distinta ed è già in forma canonica.

Una volta ottenuta la forma canonica dei turni, il passo successivo è trasformare questa lista di svolte in un singolo numero intero univoco chiamato **chiave**:

1. Una proteina composta da $N$ residui ha esattamente $N - 2$ turni interni.

2. Ciascun turno viene associato a una cifra in base 3 (FORWARD = 0, LEFT = 1, RIGHT = 2).

3. L'elenco delle cifre viene interpretato come un numero espresso in base 3, letto da sinistra verso destra. Il primo turno della catena (sullo snodo del residuo 1) rappresenta la cifra più significativa con peso $3^{N-3-i}$, mentre l'ultimo turno rappresenta la cifra delle unità.

Per comprendere il procedimento con un esempio concreto, consideriamo una catena di $N = 7$ residui con la sequenza di turni `[FORWARD, RIGHT, RIGHT, LEFT, FORWARD]` (cifre `0 2 2 1 0`):

- **Conversione in forma canonica:** Il primo turno diverso da FORWARD (all'indice 1) è un RIGHT (cifra 2). Per renderla canonica, si invertono tutte le cifre 1 e 2, ottenendo la catena riflessa `[FORWARD, LEFT, LEFT, RIGHT, FORWARD]` (cifre `0 1 1 2 0`).

- **Calcolo della chiave:** Con 5 turni, i pesi delle posizioni in base 3 sono rispettivamente $3^4 = 81$, $3^3 = 27$, $3^2 = 9$, $3^1 = 3$ e $3^0 = 1$. Moltiplicando ogni cifra canonica per il proprio peso e sommando i contributi si ottiene:

  $$(0 \cdot 81) + (1 \cdot 27) + (1 \cdot 9) + (2 \cdot 3) + (0 \cdot 1) = 0 + 27 + 9 + 6 + 0 = 42$$

  La chiave canonica risulta sempre essere il valore minimo tra la chiave della catena originaria (che sarebbe stata 75) e la chiave della sua riflessa (42), proprio perché la convenzione impone la cifra 1 anziché 2 al primo cambio di direzione.

Dalla chiave 42 è possibile fare il percorso inverso e ricostruire la sequenza dei turni dividendo ripetutamente per 3 e annotando i resti:

- $42 = 14 \cdot 3 + \mathbf{0}$

- $14 = 4 \cdot 3 + \mathbf{2}$

- $4 = 1 \cdot 3 + \mathbf{1}$

- $1 = 0 \cdot 3 + \mathbf{1}$

- $0 = 0 \cdot 3 + \mathbf{0}$

Letti dall'ultimo resto ottenuto fino al primo, i resti ricostruiscono esattamente le cifre canoniche `0 1 1 2 0`.

La chiave viene memorizzata in memoria come intero a 64 bit senza segno (`uint64`), il cui valore massimo registrabile è $2^{64} - 1 \approx 1,8 \times 10^{19}$. Siccome la chiave più grande possibile per $N - 2$ turni in base 3 è $3^{N-2} - 1$, il limite a 64 bit permette di rappresentare in modo univoco qualsiasi catena fino a $N = 42$ residui ($3^{40} \approx 1,2 \times 10^{19}$). L'utilizzo di un singolo intero a 64 bit rende le operazioni di confronto tra stati e la ricerca all'interno di tabelle o grafi estremamente veloci ed efficienti a livello di codice C++.

In sintesi, la forma canonica e la chiave servono a garantire che ogni conformazione fisica tridimensionale corrisponda a un unico ID numerico, consentendo al programma di riconoscere immediatamente gli stati già visitati, eliminare i duplicati nel dataset e costruire il grafo delle mosse.

### 5.3 Il reticolo libero

Il reticolo di `[lattice]` ha dei bordi, e i bordi cambiano le mosse possibili: vicino a una parete alcune mosse sono vietate. Così la stessa forma avrebbe mosse possibili diverse a seconda di dove si trova sulla griglia.

Per evitarlo, ogni stato del dataset viene ricostruito **al centro di un reticolo di lato N + 4** (`free_lattice_side`). Nessuna mossa sposta un residuo di più di due nodi fuori dal rettangolo che contiene la catena, quindi i bordi non vengono mai toccati: mosse ed etichette sono quelle di un reticolo infinito.

Il reticolo di `[lattice]` resta usato solo dalla REMC.

---

## 6. Le azioni: una numerazione fissa delle mosse

### 6.1 Perché serve

I modelli di apprendimento automatico (come le reti neurali) richiedono un layer di output di dimensione fissa e prefissata, nel quale la rete restituisce tipicamente un vettore di probabilità distribuito sulle azioni possibili. Per poter addestrare la rete, ciascuna posizione dell'output (chiamata **slot**) deve conservare sempre e rigorosamente lo stesso identico significato semantico in qualsiasi conformazione si trovi la proteina. Ad esempio, lo slot con indice 17 deve rappresentare in modo univoco la medesima mossa sullo stesso residuo sia che la catena sia completamente srotolata, sia che sia quasi giunta allo stato nativo.

Nelle simulazioni geometriche tradizionali su reticolo, le mosse fisicamente consentite variano continuamente ad ogni passo: la forma locale della catena determina quali spostamenti sono validi e quali sono bloccati dal momento che le caselle di arrivo potrebbero essere già occupate. Se si generasse un elenco dinamico di mosse la cui lunghezza e il cui ordine cambiassero ad ogni stato, la rete neurale non avrebbe un riferimento stabile per associare i propri neuroni di uscita alle azioni reali.

L'adozione degli **slot** risolve questa discrepanza creando uno spazio delle azioni globale e prefissato.

### 6.2 I tipi di mossa

Ogni slot è una coppia **(residuo, tipo di mossa)**. 

Si vuole analizzare nel dettaglio quali sono i tipi di mossa che, combinati con ciascun residuo della catena, vanno a comporre i singoli slot dello spazio delle azioni.

Il concetto fondamentale di questo punto è che le mosse **non sono definite tramite direzioni assolute della griglia** (come "su", "giù", "destra", "sinistra"), bensì **in modo relativo rispetto all'orientamento locale della catena proteica**. Termini come "sinistra" o "destra" sono sempre riferiti alla direzione del legame tra i residui della catena. Questo approccio garantisce due proprietà essenziali:

- **Invarianza per rotazione:** Ruotare l'intera proteina sulla griglia non modifica il nome né la classificazione delle mosse, mantenendo invariata l'azione rappresentata dallo slot.

- **Simmetria per riflessione:** Specchiare la conformazione equivale semplicemente a scambiare le mosse verso sinistra con quelle verso destra.

| Tipo                                   | Su quali residui  | Cosa fa                                                                                                                                                                                                                                                                          |
| -------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `END_L`, `END_R`, `END_B`              | primo e ultimo    | **End move VSHD**: l'estremo ruota attorno al suo unico vicino, verso sinistra, verso destra o all'indietro.                                                                                                                                                                     |
| `CORNER`                               | interni (1 … N−2) | **Corner move**: il residuo, che sta su un angolo, passa all'angolo opposto del quadratino.                                                                                                                                                                                      |
| `CRANKSHAFT`                           | 1 … N−3           | **Crankshaft** della coppia (k, k+1): i due residui di una piega a U vengono ribaltati dall'altra parte.                                                                                                                                                                         |
| `PULL_NEXT_L`, `PULL_NEXT_R`           | interni           | **Pull move** del residuo k accanto al vicino k+1, sul lato sinistro o destro del legame k → k+1. Il resto della catena viene trascinato dal lato di k−1.                                                                                                                        |
| `PULL_PREV_L`, `PULL_PREV_R`           | interni           | Come sopra, ma accanto al vicino k−1; il resto viene trascinato dal lato di k+1.                                                                                                                                                                                                 |
| `PULL_END_FF` … `PULL_END_RR` (9 tipi) | primo e ultimo    | **Pull move di un estremo**: il vicino dell'estremo va in una casella A e l'estremo in una casella B accanto ad A. La prima lettera dice dove sta A (avanti, sinistra o destra, visto dall'estremo); la seconda dove sta B (visto da A). Il resto della catena viene trascinato. |

Grazie a questa categorizzazione relativa, ogni mossa geometricamente valida viene associata a uno e un solo tipo, garantendo la corrispondenza univoca con uno slot del vettore di output.

### 6.3 Quanti sono gli slot

- Si vuole stabilire la formula matematica per calcolare il numero totale di slot (cioè la dimensione fissa del vettore delle azioni) necessari per descrivere una catena proteica di $N$ residui.

  Il calcolo dello spazio totale delle azioni si ottiene sommando i possibili tipi di mossa associati a ciascuna posizione lungo la catena:
  - **I due estremi (primo e ultimo residuo):** Ciascun estremo dispone di 12 slot dedicati (3 per le mosse VSHD di tipo `END` e 9 per le relative `PULL` dell'estremo).

  - **I residui interni (da $1$ a $N - 2$):** Ciascun residuo interno possiede 5 slot (1 per la mossa `CORNER` e 4 per le relative `PULL` interne).

  - **Le coppie interne (da $1$ a $N - 3$):** Ciascun residuo fino al penultimo-secondo aggiunge 1 slot dedicato al tipo `CRANKSHAFT`.

  Sommando questi contributi per una sequenza generica di $N$ residui si ottiene la formula complessiva:

  $$
  (2 \times 12) + ((N - 2) \times 5) + (N - 3) = 24 + 5N - 10 + N - 3 = 6N + 11 \text{ slot}
  $$

  Nel caso specifico del dataset di default con proteine da $N = 16$ residui, il calcolo porta a $6(16) + 11 = 107$ slot totali:

  - 24 slot riservati agli estremi ($2 \times 12$);

  - 70 slot riservati ai 14 residui interni ($14 \times 5$);

  - 13 slot riservati alle 13 coppie $k, k+1$ per i crankshaft.

  Gli slot vengono numerati sequenzialmente residuo per residuo, e la mappa esatta tra l'indice numerico di ciascuno slot e la coppia (residuo, tipo di mossa) è salvata nel file `meta.json`. Questa organizzazione rigida consente alla rete neurale di apprendere la scelta dell'azione secondo due modalità flessibili: selezionando direttamente una mossa tra i 107 slot disponibili, oppure effettuando una scelta a due livelli (selezionando prima su quale residuo agire e successivamente quale mossa applicargli).
- ogni residuo interno ha 5 slot: corner + 4 pull move;
- i residui da 1 a N−3 hanno in più lo slot del crankshaft.

#### Esempio

Per una proteina con $N = 16$ residui, il vettore delle azioni è un array unidimensionale di **107 elementi** (slot), indicizzati da 0 a 106, in cui ciascuna posizione corrisponde in modo fisso a una coppia `(residuo, tipo_mossa)`.

A seconda della fase di calcolo, lo stesso vettore da 107 elementi viene rappresentato in tre modi diversi:

1. **Vettore Maschera (`mask`)**: Array binario in cui `1` indica che la mossa di quel dato slot è fisicamente eseguibile nella conformazione attuale e `0` indica che è vietata (es. casella occupata o geometria non valida).

2. **Vettore Target (`optimal_actions`)**: Array binario usato per l'addestramento in cui `1` identifica le mosse ottime (che riducono la distanza dallo stato nativo) e `0` tutte le altre.

3. **Vettore di Output della Rete Neurale**: Array di 107 numeri reali che rappresentano la distribuzione di probabilità calcolata dal modello per ciascuna mossa.

A titolo di esempio, considerando una determinata conformazione della catena, la struttura interna del vettore e la mappatura dei suoi elementi si presenta così:

```
Indice Slot | Residuo | Tipo Mossa    | Maschera (Valida?) | Target (Ottima?)
----------------------------------------------------------------------------
0           | 0       | END_L         | 0 (Non valida)     | 0
1           | 0       | END_R         | 1 (Valida)         | 0 (Sub-ottimale)
2           | 0       | END_B         | 0 (Non valida)     | 0
...         | ...     | ...           | ...                | ...
15          | 2       | CORNER        | 1 (Valida)         | 1 (Mossa ottima)
16          | 2       | PULL_NEXT_L   | 1 (Valida)         | 0 (Sub-ottimale)
...         | ...     | ...           | ...                | ...
106         | 15      | PULL_END_RR   | 0 (Non valida)     | 0
```

In Python o in un framework di Deep Learning, l'intero vettore maschera o target associato a quel singolo stato si traduce in un semplice array NumPy o Tensor:

```
# Vettore target (lunghezza 107)
target = [0, 0, 0, ..., 1, 0, ..., 0]  # ha valore 1 solo negli slot 15 (e in eventuali altre mosse ottime)
```

**E per il vettore di output della rete neurale?**
Il vettore di output della rete neurale è un array di 107 numeri reali (float), con valori compresi tra 0 e 1 in seguito all'applicazione di una funzione di attivazione come la Softmax. Ciascuna posizione $i$ del vettore rappresenta la stima di probabilità del modello che la mossa associata allo slot $i$ sia l'azione migliore da compiere nella conformazione corrente.

A differenza dei vettori binari di `mask` e `optimal_actions` (composti unicamente da 0 e 1), l'output della rete è una distribuzione continua calcolata su tutti i 107 slot. Per impedire che il modello scelga un'azione fisicamente non consentita, il vettore grezzo prodotto dalla rete viene combinato tramite un prodotto elemento per elemento con il vettore `mask`. Gli slot vietati (`mask == 0`) vedono la propria probabilità annullata, e i valori delle mosse valide vengono rinormalizzati affinché la loro somma dia 1.

Un esempio pratico della struttura del vettore di output prima e dopo l'applicazione della maschera si presenta così:

```
Indice Slot | Residuo | Tipo Mossa    | Output Rete | Maschera | Probabilità Finale
------------------------------------------------------------------------------------
0           | 0       | END_L         | 0.08        | 0        | 0.00 (Mossa vietata)
1           | 0       | END_R         | 0.15        | 1        | 0.18 (Valida)
...         | ...     | ...           | ...         | ...      | ...
15          | 2       | CORNER        | 0.62        | 1        | 0.75 (Scelta preferita)
16          | 2       | PULL_NEXT_L   | 0.06        | 1        | 0.07 (Valida)
...         | ...     | ...           | ...         | ...      | ...
106         | 15      | PULL_END_RR   | 0.02        | 0        | 0.00 (Mossa vietata)
```

- **Output Rete:** È la probabilità grezza calcolata dalla rete neurale per tutti i 107 slot, senza considerare se una mossa sia fisicamente eseguibile o vietata nella conformazione corrente.

- **Probabilità Finale:** È il valore effettivo ricalcolato dopo aver azzerato le probabilità delle mosse vietate (`mask == 0`) e aver rinormalizzato i valori delle sole mosse valide affinché la loro somma sia pari a 1.



In un framework di Deep Learning (come PyTorch o TensorFlow), l'azzeramento delle mosse non valide e la normalizzazione si gestiscono direttamente a livello di tensor:

```
# output_rete: tensor (107,) restituito dall'ultimo layer della rete
# mask: tensor binario (107,) con 1 per le mosse valide e 0 per quelle vietate

output_mascherato = output_rete * mask
probabilita_finali = output_mascherato / output_mascherato.sum()
```

L'utilizzo operativo di questo vettore si divide in due contesti differenti:

- **Durante l'addestramento:** Il vettore delle probabilità finale viene confrontato con il vettore target fornito dall'oracolo (`optimal_actions` o `soft_targets`) calcolando la funzione di perdita (ad esempio la *Cross-Entropy Loss*), spingendo la rete ad aumentare i valori di probabilità in corrispondenza degli slot delle mosse ottime.

- **Durante il ripiegamento (inferenza):** L'algoritmo seleziona l'indice dello slot con il valore di probabilità più alto (approccio deterministico o *greedy*) oppure effettua un campionamento probabilistico tra le azioni valide. La mossa corrispondente a tale indice viene quindi applicata alla catena proteica per farla avanzare di un passo verso la conformazione nativa.

### 6.4 Corrispondenza con le mosse già implementate

Lo spazio delle azioni **non riscrive** la geometria delle mosse. La funzione C++ `action_moves()`:

1. chiama `vshd_moves()` e `pull_moves()` su ogni residuo, cioè usa esattamente le mosse di fase 1;
2. riconosce di che tipo è ogni mossa restituita (`classify_move()`) e la mette nel suo slot;
3. si ferma con un errore se due mosse diverse finissero nello stesso slot.

Il codice di fase 1 trova due volte solo due mosse, e in entrambi i casi è la stessa mossa:

- il crankshaft, trovato partendo da ciascuno dei due residui che sposta;
- la corner move, trovata sia tra le mosse VSHD sia tra le pull moves (la pull move che "finisce subito" è proprio una corner move).

Il risultato è una **corrispondenza 1:1**: ogni mossa del move set ha esattamente uno slot, e ogni slot contiene al più una mossa. È verificata dal test `test_action_moves_are_the_move_set`.

### 6.5 La maschera

In una data conformazione molti slot sono vuoti: la mossa di quel tipo non è possibile, perché la casella di arrivo è occupata o perché la forma locale non lo permette. La **maschera** dice quali slot sono pieni. Servirà al modello per azzerare la probabilità delle mosse impossibili.

La maschera è calcolata dal codice C++ che genera le mosse, che controlla già che le caselle di arrivo siano libere. Il test `test_mask_marks_exactly_the_moves_that_keep_a_valid_chain` applica ogni mossa della maschera e verifica che la catena resti connessa e senza sovrapposizioni.

---

## 7. Le etichette: l'oracolo

### 7.1 Perché non si usano le mosse della REMC

La REMC è un metodo Monte Carlo: sceglie le mosse a caso e ne accetta anche di sbagliate, cioè in salita di energia. Le sue traiettorie sono piene di giri inutili. Copiare le sue mosse insegnerebbe al modello a muoversi a caso.

Le etichette vengono invece da un **oracolo**, che per ogni stato sa calcolare quante mosse mancano davvero allo stato nativo.

### 7.2 Come funziona

1. Si costruisce il **grafo delle mosse**: un nodo per ogni conformazione, una freccia per ogni mossa.
2. Si parte dagli stati nativi (distanza 0) e si procede all'indietro con una **BFS**: tutti gli stati da cui si arriva a un nativo in una mossa hanno distanza 1, quelli da cui si arriva a uno stato a distanza 1 hanno distanza 2, e così via.
3. Per ogni stato e ogni slot si salva la distanza dello stato in cui porta quella mossa (`next_distance`).
4. Le **mosse ottime** di uno stato a distanza d sono quelle con `next_distance = d − 1`.

### 7.3 Lo stato nativo è l'insieme di tutti i minimi

Nel modello HP la stessa energia minima si ottiene spesso con molte conformazioni diverse. La sequenza di default, per esempio, ha **12** conformazioni canoniche diverse a energia −7. Nel dataset di default il numero varia da 1 a 9787 a seconda della sequenza (mediana 45).

Se come bersaglio si prendesse solo "la conformazione migliore trovata dalla REMC", tutte le altre conformazioni a energia minima risulterebbero non native. Il modello imparerebbe così ad allontanarsi da una conformazione che invece è già ottima. Per questo **lo stato nativo è l'insieme di tutte le conformazioni a energia minima**.

### 7.4 Criterio del percorso: il numero di mosse

La distanza conta **le mosse**, non l'energia. Le alternative considerate:

| Criterio                                                                   | Vantaggi                                                                                   | Svantaggi                                                                                                          |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------ |
| **Numero di mosse** (scelto)                                               | Non ha parametri, dà etichette esatte, e l'insieme delle mosse ottime viene fuori da solo. | Non tiene conto delle barriere di energia: il percorso più corto può passare per stati ad alta energia.            |
| Discesa d'energia (scegliere sempre la mossa che abbassa di più l'energia) | Intuitivo.                                                                                 | Il paesaggio energetico HP è pieno di minimi locali: la discesa si ferma lì e spesso non arriva allo stato nativo. |
| Costo misto: 1 per mossa + λ per ogni unità di energia in salita           | Percorsi più "fisici".                                                                     | Il risultato dipende da λ, scelto a mano; ci sono meno mosse ottime e le etichette sono meno stabili.              |

Per non perdere l'informazione sull'energia, ogni record salva anche **il cambio di energia di ogni mossa** (`next_energy_delta`). Chi vuole favorire le mosse che scendono in energia può farlo al caricamento (§12.2), senza rigenerare il dataset.

### 7.5 Due versioni dell'oracolo

- **Oracolo esatto**, per catene fino a `exact_max_length` residui (di default 16).
  - Il grafo contiene **tutte** le conformazioni possibili, generate da `enumerate_states`. Per N = 16 sono 802 075 stati e 37,6 milioni di frecce; costruire il grafo richiede circa 5 secondi.
  - Il grafo non dipende dalla sequenza: si costruisce una volta sola e si riusa per tutte le sequenze. Per ogni sequenza cambiano solo le energie, e quindi gli stati nativi (circa 1,5 secondi).
  - Le distanze sono esatte e ogni stato ha una distanza finita, perché le pull moves permettono di raggiungere qualsiasi conformazione.
- **Oracolo sul grafo REMC**, per catene più lunghe, dove tutte le conformazioni sarebbero troppe.
  - Il grafo contiene solo gli stati visitati dalla REMC.
  - Le frecce non sono solo le mosse che la REMC ha fatto davvero: si aggiunge una freccia tra **qualsiasi** coppia di stati visitati che distano una mossa. Queste scorciatoie accorciano molto i percorsi.
  - Le distanze sono **limiti superiori**: il percorso vero più corto potrebbe passare per stati che la REMC non ha visto.
  - Gli stati nativi sono gli stati visitati con l'energia più bassa trovata.

### 7.6 Il grafo delle mosse ha frecce a senso unico

Costruendo il grafo è emerso che **alcune mosse non si possono annullare con una sola mossa**. Succede con due pull move degli estremi, `PULL_END_LL` e `PULL_END_RR`.

Esempio: una catena dritta, con una di queste mosse, diventa un "uncino" (l'estremo ripiegato all'indietro). Dall'uncino però nessuna mossa singola riporta la catena dritta. Il motivo è che il trascinamento della pull move si ferma appena un residuo è già accanto al precedente, e nel verso opposto questo succede prima.

Conseguenze:

- **per il dataset nessun problema**: la BFS usa le frecce invertite, calcolate esplicitamente, senza supporre che ogni mossa abbia la sua inversa (test `test_the_move_graph_is_directed`);
- **per la fase 1 è un'osservazione da tenere presente**: il criterio di Metropolis campiona esattamente la distribuzione di Boltzmann solo se le mosse proposte sono reversibili. La REMC quindi la campiona solo in modo approssimato. Il codice di fase 1 non è stato modificato.

---

## 8. Le traiettorie REMC e la loro pulizia

### 8.1 Registrare le traiettorie

La REMC fa evolvere più repliche a temperature diverse e ogni tanto scambia le conformazioni tra due temperature vicine. Uno scambio non cambia la forma di una conformazione, ne cambia solo la temperatura. Seguendo una stessa conformazione attraverso gli scambi (un **walker**) si ottiene quindi una catena continua di mosse singole: la traiettoria del walker.

Durante la simulazione un oggetto C++, il `SampleRecorder`, salva:

- **la traiettoria di ogni walker**: la chiave dello stato iniziale e quella dopo ogni mossa accettata, fino a `max_recorded_steps` mosse per walker;
- **uno snapshot di ogni replica** ogni `save_every` cicli: sono i decoy, raccolti a tutte le temperature.

La registrazione non usa il generatore di numeri casuali, quindi la simulazione è identica con o senza registrazione (test `test_recording_does_not_change_the_simulation`). I dati restano in memoria e i file li scrive Python, come per il resto del progetto, dove il C++ non legge e non scrive file.

### 8.2 I tre passi di pulizia

Una traiettoria Monte Carlo fa molti giri inutili: dove basterebbe andare da A a D, tipicamente fa A → B → C → B → A → E → D. Ogni traiettoria viene tagliata al primo stato nativo raggiunto e poi accorciata in tre passi:

1. **Rimozione dei cicli** (loop truncation). Se la traiettoria torna in uno stato già visitato, tutto quello che sta in mezzo si cancella. A → B → C → B → A → E → D diventa A → E → D. Per riconoscere che due stati sono lo stesso si usa la chiave canonica (§5.2).
2. **Cammino più corto tra gli stati visitati.** Si costruisce un grafo che contiene *solo* gli stati visitati da quella traiettoria, ma con *tutte* le mosse possibili tra loro, anche quelle che la REMC non ha mai fatto. Se A e D distano una sola mossa, il cammino diventa A → D.
3. **Confronto con i grafi più grandi:** il cammino più corto nel grafo di tutti i walker della sequenza e, quando c'è l'oracolo esatto, la distanza vera.

### 8.3 Lunghezze prima e dopo la pulizia

Per misurare l'effetto di ogni passo, il file `trajectories.csv` salva per ogni walker:

| Colonna              | Cosa misura                                                          |
| -------------------- | -------------------------------------------------------------------- |
| `raw_length`         | mosse fatte dalla REMC fino al primo stato nativo                    |
| `loop_erased_length` | mosse dopo la rimozione dei cicli                                    |
| `shortest_length`    | mosse del cammino più corto tra gli stati visitati da quel walker    |
| `graph_length`       | mosse del cammino più corto tra gli stati visitati da tutti i walker |
| `exact_length`       | distanza vera (solo con l'oracolo esatto; altrimenti −1)             |

Ogni passo può solo accorciare, quindi vale sempre `raw ≥ loop_erased ≥ shortest ≥ graph ≥ exact` (verificato dai test).

**Risultati sul dataset di default** (489 walker su 500 hanno raggiunto lo stato nativo):

|                 | grezza | senza cicli | cammino più corto (walker) | cammino più corto (tutti i walker) | esatta |
| --------------- | ------ | ----------- | -------------------------- | ---------------------------------- | ------ |
| lunghezza media | 806    | 311         | 66,5                       | 9,6                                | 5,3    |

Per la sequenza di default, il walker 0 passa da 1019 → 786 → 61 → 12 → 6 mosse.

### 8.4 Come entrano nel dataset

Le traiettorie ripulite diventano percorsi di tipo `remc_pruned`. I loro stati vengono etichettati dall'oracolo come tutti gli altri: le mosse della traiettoria non sono necessariamente ottime, ma le etichette di ogni stato sì. I percorsi decoy → nativo della §9 sono lo stesso procedimento portato all'ultimo livello: il cammino più corto nel grafo più grande disponibile.

---

## 9. Come viene generato

### 9.1 Le sequenze

- La prima è `protein.sequence` della configurazione (se `include_config_sequence = true`).
- Le altre sono casuali, di lunghezza `protein.length`, con una probabilità `h_fraction` che ogni residuo sia H.
- **Una sequenza e la sua inversa** (letta dall'ultimo residuo) descrivono la stessa catena: se ne tiene una sola. Altrimenti lo stesso problema potrebbe finire sia nei dati di addestramento sia in quelli di test.
- Si scartano le sequenze con energia minima 0: ogni stato sarebbe nativo e non ci sarebbe nulla da imparare.

### 9.2 I passi per ogni sequenza

1. **REMC** con i parametri di `[dataset.remc]`, partendo dalla catena dritta. Le temperature arrivano fino a 1,0 (contro 0,22 della fase 1), così tra gli snapshot ci sono anche conformazioni srotolate e non solo compatte.
2. **Oracolo** (esatto o sul grafo REMC) e calcolo degli stati nativi.
3. **Pulizia** delle traiettorie dei walker (§8).
4. **Decoy:** per ogni replica, cioè per ogni temperatura, si scelgono a caso `decoys_per_temperature` snapshot diversi.
5. **Percorsi ottimi:** da ogni decoy si segue una mossa ottima alla volta fino allo stato nativo. Quando ci sono più mosse ottime se ne sceglie una a caso, con un seed fissato. Nel dataset entrano **tutti** gli stati del percorso, non solo quello di partenza.
6. **Stati di recupero:** una frazione `perturb_fraction` degli stati dei percorsi viene spostata con 1 … `perturb_max_moves` mosse casuali. Sono stati "fuori strada", simili a quelli in cui finirà il modello dopo un errore, e l'oracolo dice come rientrare.
7. **Etichette:** si eliminano i doppioni (stessa chiave) e si calcolano le etichette di tutti gli stati in un'unica chiamata C++.

---

## 10. I tre problemi da evitare e come sono affrontati

### 10.1 Accumulo dell'errore

*Il problema:* durante l'uso il modello sbaglia qualche mossa e finisce in stati che non ha mai visto nei dati; da lì sbaglia ancora di più.

*Cosa fa il dataset:*

- raccoglie stati a tutte le temperature, dai più srotolati ai quasi nativi;
- etichetta tutti gli stati lungo i percorsi, non solo quelli di partenza;
- aggiunge gli **stati di recupero** (§9.2, punto 6);
- salva la distanza dopo **ogni** mossa, non solo dopo quella ottima. Il modello può così imparare anche quanto è "sbagliata" una mossa, non solo quale sia quella giusta;
- con l'oracolo esatto si possono etichettare in qualunque momento anche stati nuovi. In fase 3 si potranno aggiungere al dataset gli stati in cui il modello finisce davvero sbagliando.

### 10.2 REMC casuale contro modello deterministico

*Il problema:* le traiettorie REMC sono casuali e piene di mosse inutili o in salita.

*Cosa fa il dataset:* la REMC fornisce **solo gli stati** (e, per catene lunghe, le frecce del grafo). Le mosse giuste vengono sempre dall'oracolo.

### 10.3 Più mosse ugualmente buone

*Il problema:* da uno stesso stato possono esserci più mosse ottime; sceglierne una a caso come "la risposta" insegnerebbe al modello cose contraddittorie.

*Cosa fa il dataset:* le conserva **tutte**. Dalla distanza dopo ogni mossa si ricava l'insieme completo delle mosse ottime (§12.2). Nel train del dataset di default il 69% degli stati non nativi ha più di una mossa ottima, in media 4,9 su 41 mosse possibili.

---

## 11. I file

### 11.1 Elenco

| File                               | Contenuto                                                                                                                                                                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `meta.json`                        | Descrizione del dataset: lunghezza N, numero di slot, nomi dei tipi di mossa, tabella degli slot, permutazione per lo specchio, codici di turni e sorgenti, tipo di oracolo, numero di record per split, configurazione usata, commit git. |
| `sequences.npz`                    | Una riga per sequenza: la sequenza, la sua versione 0/1 (`hp`), lo split, l'energia minima, il numero di stati nativi, l'energia migliore trovata dalla REMC, se l'oracolo era esatto, quanti stati ha visitato la REMC.                   |
| `train.npz`, `val.npz`, `test.npz` | I record e i percorsi di ogni split (§11.2).                                                                                                                                                                                               |
| `trajectories.csv`                 | Le lunghezze delle traiettorie REMC prima e dopo la pulizia (§8.3).                                                                                                                                                                        |

Per leggerli basta numpy.

### 11.2 Campi di un record

S è il numero di record dello split, A = 6N + 11 il numero di slot.

| Campo               | Tipo    | Forma    | Contenuto                                                                                                                          |
| ------------------- | ------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `seq_id`            | int32   | (S,)     | numero della sequenza in `sequences.npz`                                                                                           |
| `turns`             | uint8   | (S, N−2) | turni della forma canonica (0 = FORWARD, 1 = LEFT, 2 = RIGHT)                                                                      |
| `energy`            | int16   | (S,)     | energia della conformazione                                                                                                        |
| `distance`          | int16   | (S,)     | mosse che mancano allo stato nativo (0 = nativo)                                                                                   |
| `mask`              | bit     | (S, A)   | slot possibili; sul disco 8 slot per byte, `load_split` li restituisce come vero/falso                                             |
| `next_distance`     | int16   | (S, A)   | distanza dopo la mossa di ogni slot; **−1** = mossa impossibile, **−2** = distanza sconosciuta (solo con l'oracolo sul grafo REMC) |
| `next_energy_delta` | int8    | (S, A)   | cambio di energia di ogni mossa                                                                                                    |
| `source`            | uint8   | (S,)     | da dove viene il record: `remc_snapshot`, `remc_pruned`, `oracle_path`, `perturbed`                                                |
| `temperature`       | float32 | (S,)     | temperatura dello snapshot REMC (vuoto, NaN, per gli altri)                                                                        |

Le mosse ottime non sono salvate a parte: si ricavano sempre da `next_distance`, così l'informazione ha una sola fonte.

### 11.3 I percorsi

Ogni split contiene anche i percorsi verso lo stato nativo, salvati come elenchi di numeri di record:

- `path_records[path_offsets[p] : path_offsets[p+1]]` sono i record del percorso p, in ordine;
- `path_actions` ha la stessa lunghezza: l'elemento i è lo slot della mossa che porta al record successivo (−1 sull'ultimo record);
- `path_kind` dice se il percorso è ottimo (`oracle`) o una traiettoria REMC ripulita (`remc_pruned`);
- `path_seq_id` dice a quale sequenza appartiene.

Ogni percorso finisce in uno stato nativo (test `test_every_path_ends_in_the_native_state`).

---

## 12. Come si usa

### 12.1 Lettura e codifiche

```python
from hpfold import dataset

data = dataset.load_split("data/output/dataset", "train")
meta = dataset.load_meta("data/output/dataset")

coords = dataset.turns_to_coordinates(data["turns"])   # coordinate (S, N, 2)
contacts = dataset.contact_map(coords)                 # matrice di contatto (S, N, N)
grid = dataset.lattice_grid(coords, data["hp"])        # immagine del reticolo (S, 3, N, N)
```

I turni sono la codifica "a mosse relative"; da questi si ottengono coordinate, matrice di contatto e griglia. La griglia ha tre canali: residui H, residui P e posizione lungo la catena.

### 12.2 Le risposte giuste (target)

```python
target = dataset.optimal_actions(data["next_distance"], data["distance"])
```

restituisce, per ogni record, vero sugli slot delle mosse ottime (più di uno può essere vero). In alternativa:

- `dataset.soft_targets(next_distance, distance)` dà una distribuzione uniforme sulle mosse ottime;
- aggiungendo `next_energy_delta` e una `temperature`, le mosse ottime che abbassano l'energia pesano di più.

Gli stati nativi (distanza 0) non hanno mosse ottime. Se il modello deve anche capire quando fermarsi, `distance == 0` può fare da etichetta "stop".

### 12.3 Simmetrie

Ogni stato è salvato **una sola volta**, in forma canonica. Salvare anche le copie ruotate sarebbe inutile: con turni e mosse relative una rotazione non cambia nulla, e le copie sarebbero righe identiche.

Le simmetrie si applicano invece al caricamento:

- **Specchio:** `dataset.augment(batch, meta["mirror_permutation"], rng)` specchia metà dei record a caso. Scambia LEFT e RIGHT nei turni e sposta gli slot delle mosse secondo `mirror_permutation` (per esempio `PULL_NEXT_L` ↔ `PULL_NEXT_R`). Energia e distanza non cambiano. **Questo passo è necessario:** senza, il modello vedrebbe solo catene il cui primo turno è a sinistra.
- **Rotazione:** serve solo alla codifica a griglia. `dataset.rotate_coordinates` ruota le coordinate di 90°, e le etichette restano le stesse.

Il test `test_rotations_keep_the_actions_and_reflections_permute_them` controlla che ruotando e specchiando davvero la catena, e ricalcolando le mosse in C++, si ottenga esattamente quello che prevedono le due regole.

---

## 13. Divisione in train, validation e test

La divisione è **per sequenza**: tutti i record e i percorsi di una sequenza finiscono nello stesso insieme (proporzioni `split = [0.8, 0.1, 0.1]`, con un seed). Così il test misura quanto il modello sa ripiegare sequenze che non ha mai visto, e non solo stati nuovi di sequenze già note.

La stessa conformazione può comparire in insiemi diversi con sequenze diverse. Non è un problema, perché l'input del modello è la coppia (sequenza, conformazione).

---

## 14. Parametri (`config/default.toml`)

### `[dataset]`

| Parametro                 | Default         | Significato                                                                                                  |
| ------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------ |
| `n_sequences`             | 100             | numero di sequenze                                                                                           |
| `h_fraction`              | 0.5             | probabilità che un residuo di una sequenza casuale sia H                                                     |
| `include_config_sequence` | true            | usa `protein.sequence` come prima sequenza                                                                   |
| `seed`                    | 7               | seme per sequenze, split e scelta degli stati                                                                |
| `split`                   | [0.8, 0.1, 0.1] | proporzioni di train, validation e test                                                                      |
| `exact_max_length`        | 16              | lunghezza massima con l'oracolo esatto. Il numero di stati cresce di circa 2,7 volte per ogni residuo in più |
| `decoys_per_temperature`  | 20              | stati di partenza presi da ogni replica                                                                      |
| `perturb_fraction`        | 0.25            | stati di recupero, come frazione degli stati dei percorsi                                                    |
| `perturb_max_moves`       | 3               | massimo numero di mosse casuali per creare uno stato di recupero                                             |
| `max_recorded_steps`      | 50000           | mosse registrate al massimo per ogni walker                                                                  |

### `[dataset.remc]`

La REMC lanciata per ogni sequenza: stesse chiavi di `[remc]` (`n_replicas`, `t_min`, `t_max`, `local_steps`, `n_cycles`, `rho`, `seed`), più `save_every`, cioè ogni quanti cicli salvare uno snapshot. È più corta della REMC di fase 1 e arriva a temperature più alte. La sequenza i usa il seme `seed + i`.

---

## 15. Il dataset di default in numeri

|                                                  |                                                                                                    |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| sequenze                                         | 100 di 16 residui (80 train, 10 validation, 10 test)                                               |
| oracolo                                          | esatto (802 075 stati, 37,6 milioni di frecce)                                                     |
| record                                           | 75 590 (60 739 train, 7 333 validation, 7 518 test)                                                |
| percorsi                                         | 10 313 (9 824 ottimi, 489 traiettorie REMC ripulite)                                               |
| distanza massima dallo stato nativo tra i record | 8 mosse                                                                                            |
| mosse possibili per stato                        | 41 in media, su 107 slot                                                                           |
| stati non nativi con più di una mossa ottima     | 69% (4,9 mosse ottime in media)                                                                    |
| tipo dei record (train)                          | 7 672 snapshot REMC, 25 635 da traiettorie ripulite, 15 676 da percorsi ottimi, 11 756 di recupero |
| tempo di generazione                             | circa 3 minuti e 40 secondi                                                                        |
| memoria usata                                    | circa 640 MB                                                                                       |
| spazio su disco                                  | 3,5 MB                                                                                             |

Le distanze sono piccole perché una pull move sposta molti residui in una volta. Per la sequenza di default, per esempio, nessuna delle 802 075 conformazioni dista più di 7 mosse da uno stato nativo. Le righe su mosse possibili, mosse ottime e tipo dei record sono calcolate sullo split di train.

---

## 16. Dove sta il codice

Le parti che lavorano su milioni di stati sono in C++; le scelte e i file sono in Python.

| File                                                     | Contenuto                                                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| [canonical.hpp](../cpp/include/hp/canonical.hpp)         | turni, forma canonica, chiave, reticolo libero (§5)                                                        |
| [actions.hpp](../cpp/include/hp/actions.hpp)             | slot, classificazione delle mosse, permutazione per lo specchio (§6)                                       |
| [state_graph.hpp](../cpp/include/hp/state_graph.hpp)     | elenco di tutte le conformazioni, grafo delle mosse, oracolo, percorsi, pulizia delle traiettorie (§7, §8) |
| [sample_writer.hpp](../cpp/include/hp/sample_writer.hpp) | `SampleRecorder`: registrazione delle traiettorie e degli snapshot REMC (§8.1)                             |
| [remc.hpp](../cpp/include/hp/remc.hpp)                   | `run_remc` accetta un `SampleRecorder` facoltativo                                                         |
| [bindings.cpp](../cpp/bindings/bindings.cpp)             | espone tutto a Python; i risultati grandi arrivano come array numpy                                        |
| [dataset.py](../python/hpfold/dataset.py)                | sequenze, scelta degli stati, split, scrittura e lettura dei file, codifiche, simmetrie (§9, §11, §12)     |
| [test_dataset.py](../tests/test_dataset.py)              | i test (§17)                                                                                               |

Le note sui singoli file sono in `notes/file_functioning/cpp/include/hp/` e in `notes/python/hpfold/dataset.md`.

---

## 17. Cosa verificano i test

`tests/test_dataset.py` controlla:

- **azioni:** il numero e l'ordine degli slot; la permutazione per lo specchio scambia davvero sinistra e destra; ogni mossa di fase 1 ha esattamente uno slot; ogni mossa della maschera lascia una catena valida;
- **forma canonica:** la chiave non cambia con rotazioni, riflessioni e traslazioni; il numero di conformazioni elencate coincide con quello noto (2, 13, 272, 2034 per N = 3, 5, 8, 10); le coordinate ricavate in Python coincidono con quelle del C++; la matrice di contatto ridà l'energia;
- **simmetrie:** ruotare la catena non cambia le mosse, specchiarla le permuta come previsto; specchiare un record due volte lo riporta com'era;
- **oracolo:** le distanze rispettano la definizione di cammino più corto; il grafo ha davvero frecce a senso unico; i percorsi finiscono in uno stato nativo; le distanze del grafo REMC non sono mai più corte di quelle esatte;
- **REMC:** la registrazione non cambia la simulazione; stati consecutivi di una traiettoria distano una mossa; le lunghezze si accorciano a ogni passo di pulizia;
- **file generati** (sia con l'oracolo esatto sia con quello sul grafo REMC): le maschere coincidono con quelle del C++; ogni stato non nativo ha almeno una mossa ottima; ogni percorso termina nello stato nativo; i tre split non hanno sequenze in comune; l'augmentation specchia circa metà dei record.

---

## 18. Limiti e cose non fatte

- **Una lunghezza per dataset.** Il numero di slot dipende da N, quindi tutte le sequenze di un dataset hanno la stessa lunghezza.
- **Oltre 16 residui le distanze non sono esatte.** Con l'oracolo sul grafo REMC sono limiti superiori, e la loro qualità dipende da quanto la REMC ha esplorato.
- **La REMC non è salvata a parte.** I dati grezzi della REMC non vengono scritti in `output.samples_dir`: per rigenerare il dataset si rilancia anche la REMC. Le chiavi `remc.save_every` e `output.samples_dir` di fase 1 restano inutilizzate.
- **Bilancio dettagliato della fase 1.** Con le mosse a senso unico della §7.6, la REMC campiona la distribuzione di Boltzmann solo in modo approssimato.
- **Nessuna mossa "stop".** Lo spazio delle azioni contiene solo mosse; per decidere quando fermarsi si può usare `distance == 0` .

ptroca
