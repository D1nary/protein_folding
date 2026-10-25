# Il dataset per il decoy discriminator (fase 2b)

Questa nota raccoglie tutto quello che riguarda il dataset del *decoy discriminator*: a cosa serve, come è fatto, come viene generato, perché è stato costruito così e come si usa. Le sezioni si possono leggere in ordine; la §2 spiega le parole che ritornano in tutto il resto. Molti concetti (REMC, walker, snapshot, forma canonica, chiave) sono gli stessi del dataset per la policy: qui sono ripresi in breve, i dettagli sono in [policy_dataset.md](policy_dataset.md).

Per generarlo:

```bash
python -m hpfold --config config/default.toml discriminator_dataset
```

I file finiscono nella cartella `output.discriminator_dataset_dir` (di default `data/output/datasets/discriminator_dataset/`).

Questo dataset insegna all'algoritmo ML la seguente cosa: data una sequenza e una conformazione già fatta, il modello deve dire **se quella conformazione è uno stato nativo oppure no** (un *decoy*, cioè una forma "esca" che sembra plausibile ma non ha l'energia minima). È un classificatore binario, che in un secondo momento può fare da giudice per le conformazioni proposte da altri metodi (per esempio dalla policy).

La pipeline è **indipendente** da quella del policy dataset: ha le sue sezioni di configurazione, la sua REMC e la sua cartella di output. Da `policy_dataset.py` importa solo alcune funzioni di utilità (generazione delle sequenze, split, conversione delle chiavi in turni e coordinate, matrice di contatto).

---

## 1. Come vengono utilizzati i dati del dataset

### ML supervisionato

Si tratta di apprendimento supervisionato: durante il training del modello, ogni esempio è una coppia $(x_i, y_i)$, dove $x_i$ sono le **feature** (l'input) e $y_i$ il **target** (la risposta corretta). Durante l'addestramento il modello vede molte coppie e impara a prevedere $y$ a partire da $x$. 
Il target è **un sì/no**: nativo (1) o decoy (0). È un problema di **classificazione binaria**.

### Caso del progetto (esempio)

Ogni riga del dataset è un **esempio**: una "fotografia" di una proteina in una certa forma, con scritto dall'insegnante se quella forma è nativa o no.

**Feature (input del modello):**

- **`sequence`**: la sequenza HP della proteina (es. `HHPHHPHHPPHPPHHP`). Serve sempre insieme alla conformazione: la stessa forma può essere nativa per una sequenza e pessima per un'altra.
- **`turns`**: la conformazione come serie di svolte relative, scritta con le lettere `F` (dritto), `L` (sinistra), `R` (destra), in forma canonica. Sono $N - 2$ lettere.
- **`coordinates`**: la stessa conformazione come coordinate $(x, y)$ dei residui, già pronte. Sono in forma canonica: residuo 0 nell'origine e primo legame lungo $+x$ (le coordinate possono quindi essere negative).

Turni e coordinate descrivono la stessa forma: il modello ne userà una sola, a seconda della codifica scelta (§10.1).

**Target (risposta corretta):**

- **`label`**: **1** se la conformazione è nativa, **0** se è un decoy. È il target principale.
- **`energy`**, **`e_min`**, **`gap`**: l'energia HP della conformazione, l'energia minima della sequenza e la differenza tra le due (`gap = energy − e_min`, 0 per un nativo). Servono come **target continuo**, se invece di un sì/no si vuole che il modello dia un punteggio di stabilità (regressione sul gap, oppure ranking: "questa forma è più stabile di quella").

**Informazioni in più (non vanno date in input al modello):**

- **`tier`**: la difficoltà del decoy, `near`, `mid` o `far` (§5.2); `native` per i nativi.
- **`kinetic_trap`**: 1 se il decoy è una trappola cinetica, cioè un minimo locale dell'energia (§5.3).
- **`native_overlap`**: quanti contatti H-H del nativo sono presenti anche in questa forma (§5.4).
- **provenienza REMC**: `source`, `walker`, `replica`, `temperature`, `cycle`, `step`, `key` (§9.2).

Questi campi **non devono finire tra le feature**: `tier`, `gap` e `energy` contengono di fatto la risposta (un esempio con `gap = 0` è nativo per definizione). Servono per analizzare gli errori ("il modello sbaglia soprattutto sui decoy `near`?"), per pesare gli esempi o per costruire target alternativi.

#### Un esempio

> "Sequenza n. 0 (`HHPHHPHHPPHPPHHP`), forma = `LLRLLFRLLRLLRR`, energia −7, energia minima −7 → **nativo** (label 1)."
>
> "Stessa sequenza, un'altra forma con energia −6 → **decoy** (label 0), gap 1, fascia `near`, trappola cinetica: nessuna mossa singola la fa scendere di energia."

Il secondo è un esempio difficile: è a un solo contatto dallo stato nativo e una simulazione Monte Carlo ci si potrebbe fermare. Il modello deve imparare a distinguerlo lo stesso.

### Da dove vengono gli esempi?

Il dataset parte da **`n_sequences` = 2000 sequenze HP diverse**, generate a caso (§7). Per **ognuna delle 2000 sequenze** si lancia una simulazione REMC separata, partendo dalla catena dritta: in tutto quindi 2000 simulazioni, una per sequenza. Gli esempi di una sequenza vengono tutti dalla sua simulazione. **Da ciascuna simulazione si raccolgono tutti gli stati distinti visitati** (§3), si calcola l'energia di ognuno e poi, sempre sequenza per sequenza:

1. **Nativi (label 1).** Tutti gli stati distinti con l'energia più bassa trovata, cioè **tutte le copie degeneri dello stato fondamentale** che la REMC ha incontrato, fino a un massimo di `examples_per_class` (20). Se sono di più, se ne scelgono 20 a caso (§4).
2. **Decoy (label 0).** Stati visitati con energia **strettamente maggiore** del minimo, **tanti quanti i nativi**. Sono distribuiti in tre fasce di difficoltà (`near`, `mid`, `far`) a seconda di quanto la loro energia è lontana da quella nativa; nella fascia `near` si preferiscono le trappole cinetiche (§5).

A differenza del policy dataset, qui **non c'è un oracolo**: l'etichetta di un esempio si ricava direttamente dall'energia (nativo ⇔ energia uguale a quella minima). L'unico punto delicato è essere sicuri che la REMC abbia davvero trovato l'energia minima: per catene fino a 20 residui questo si controlla con l'**enumerazione esatta** e le sequenze in cui la REMC l'ha mancata vengono scartate (§7).

### Distribuzione degli esempi (dataset di default, 62 640 esempi in tutto)

Per classe e split:

```
          nativi   decoy
train     25 177  25 177
val        3 028   3 028
test       3 115   3 115
totale    31 320  31 320
```

Decoy per fascia di difficoltà:

```
near   ████████████████████▌     38,6 %   (12 095)
mid    ████████████████▉         31,8 %   (9 953)
far    ███████████████▊          29,6 %   (9 272)
```

Decoy per gap dallo stato nativo:

```
+1   ████████████████████████████████████████  8951
+2   ██████████████████████████████████████    8423
+3   ██████████████████████████                5764
+4   ███████████████████                       4282
+5   █████████████                             2949
+6   ████                                       816
+7   █                                          126
+8   ▏                                            9
```

La fascia `near` è un po' più grande delle altre perché le quote che le altre fasce non riescono a riempire passano a quelle più vicine (§5.2), e perché il resto della divisione per 3 va alle fasce più vicine. I gap grandi sono rari perché sono possibili solo per le sequenze con energia minima molto negativa.

Questi numeri sono quelli di `report.txt` del dataset generato con la configurazione di default.

---

## 2. Parole usate in questa nota

| Parola                         | Significato                                                                                                                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Conformazione** / **stato**  | La forma della catena sul reticolo: dove si trova ogni residuo.                                                                                     |
| **Energia HP**                 | Meno il numero di contatti H-H: coppie di residui H vicini sul reticolo ma non consecutivi nella catena.                                            |
| **Stato nativo**               | Una conformazione con l'energia più bassa per quella sequenza. Spesso ce n'è più di una (stati **degeneri**).                                       |
| **Decoy**                      | Una conformazione non nativa usata come esempio negativo. Qui i decoy sono stati visitati dalla REMC con energia maggiore di quella minima.         |
| **Gap**                        | La differenza `energy − e_min`: quanti contatti H-H mancano rispetto allo stato nativo. 0 per un nativo.                                            |
| **Fascia** (*tier*)            | La difficoltà di un decoy, decisa dal gap relativo a $\lvert E_{min} \rvert$: `near` (vicino al nativo, difficile), `mid`, `far` (lontano, facile). |
| **Trappola cinetica**          | Un decoy da cui nessuna mossa singola fa scendere l'energia: un minimo locale in cui il Monte Carlo può restare bloccato.                           |
| **Sovrapposizione col nativo** | La frazione dei contatti H-H di uno stato nativo che compaiono anche nel decoy (`native_overlap`).                                                  |
| **Chiave canonica**            | Il numero intero che identifica una conformazione a meno di traslazioni, rotazioni e riflessioni (vedi [policy_dataset.md §5](policy_dataset.md)).  |
| **Walker** / **replica**       | Come nel policy dataset: il walker è una catena seguita per tutta la simulazione, la replica è un posto a temperatura fissa.                        |
| **Snapshot**                   | Fotografia periodica della conformazione di ogni replica.                                                                                           |
| **Traiettoria**                | Il "filmato" di un walker: lo stato dopo ogni mossa accettata.                                                                                      |

---

## 3. Cosa si prende dalla REMC

Per ogni sequenza viene lanciata una REMC con i parametri di `[discriminator_dataset.remc]`, partendo dalla catena dritta al centro del reticolo.

Prima di iniziare la spiegazione teniamo a mente la seguente distinzione tra replica e walker nell'algoritmo REMC:

- **Replica**: uno dei 5 **posti a temperatura fissa** della simulazione. La replica non cambia mai temperatura. Cambia invece la conformazione che la occupa: dopo uno scambio, nella replica fredda può esserci una catena diversa da quella di prima.
- **Walker**: una delle 5 **catene simulate**, seguita per tutta la durata della simulazione. Durante la simulazione la catena cambia temperatura ma il walker relativo rimane sempre lo stesso.



La REMC usa lo stesso `SampleRecorder` del policy dataset e registra:

- **le traiettorie dei walker**: la chiave canonica dello stato dopo ogni mossa accettata, fino a `max_recorded_steps` (50 000) mosse per walker;
- **gli snapshot**: la conformazione di ogni replica ogni `save_every` (2) **cicli REMC**, con la temperatura, il ciclo e il walker che la occupava.

Un **ciclo REMC** è l'unità di base della simulazione ed è fatto di due parti:

1. **passi locali**: ogni replica esegue `local_steps` (100) passi Monte Carlo Metropolis alla propria temperatura (a ogni passo si propone una mossa, pull move con probabilità `rho` oppure VSHD, e la si accetta o rifiuta);
2. **scambi**: si tentano gli scambi di conformazione tra repliche a temperature vicine.

La simulazione ripete il ciclo `n_cycles` (500) volte. Lo snapshot si scatta alla fine dei passi locali, prima degli scambi, in un ciclo ogni 2: quindi ogni **200 passi Monte Carlo tentati** per replica. In tutta la simulazione sono 250 snapshot per replica, 1 250 in tutto con 5 repliche. Tra due snapshot la catena ha fatto molte mosse, quindi gli snapshot sono stati sparsi e non un percorso continuo (quello lo dà la traiettoria).

Rispetto al policy dataset, qui le due registrazioni hanno lo **stesso ruolo**: sono solo due fonti di stati. Non servono percorsi, quindi le traiettorie non vengono ripulite né tagliate al primo nativo; non servono punti di partenza, quindi gli snapshot non vengono scelti per temperatura.

### 3.1 Gli stati visitati

Alla fine della REMC di una sequenza si vuole un **elenco di tutte le conformazioni diverse che la simulazione ha incontrato**, ognuna una volta sola. È da questo elenco che poi si scelgono nativi e decoy (§4, §5). Lo costruisce la funzione `visited_states`.

Il problema è che la stessa forma compare moltissime volte: un walker ci ripassa, un altro walker ci arriva per conto suo, uno snapshot la fotografa. E può comparire spostata, ruotata o specchiata sulla griglia. Per costruire l'elenco serve quindi prima un modo per dire **"queste due conformazioni sono la stessa"**: la chiave canonica.

#### 3.1.1 La chiave canonica

È la stessa usata dal policy dataset ([policy_dataset.md §5](policy_dataset.md)). Si ottiene in tre passi.

**1. Perché non usare le coordinate.** Se si descrivesse una catena con le coordinate $(x, y)$ dei residui, la stessa forma avrebbe tante descrizioni diverse: basta traslarla sulla griglia o ruotarla in uno dei 4 orientamenti e le coordinate cambiano, anche se la forma fisica è identica. Il programma tratterebbe come stati diversi quelli che sono lo stesso stato.

**2. I turni.** Invece di dire *dove* si trova ogni residuo, si percorre la catena dal primo all'ultimo residuo e, a ogni residuo interno, si annota cosa fa la catena rispetto alla direzione da cui arriva:

- **FORWARD** (cifra 0): prosegue dritta;
- **LEFT** (cifra 1): gira a sinistra;
- **RIGHT** (cifra 2): gira a destra.

Il primo e l'ultimo residuo non svoltano, quindi una catena di $N$ residui è descritta da **$N - 2$ turni**. I turni non cambiano se la catena viene traslata o ruotata: due copie della stessa forma in punti o orientamenti diversi della griglia hanno gli stessi turni.

*Esempio.* La catena `(0,0) (1,0) (2,0) (2,1) (1,1)` va dritta, poi gira due volte a sinistra: i turni sono `[FORWARD, LEFT, LEFT]`.

**3. La forma canonica.** I turni non risolvono ancora l'effetto specchio: riflettendo una conformazione, energia e geometria restano le stesse, ma ogni LEFT diventa RIGHT e viceversa. Per non avere due descrizioni dello stesso stato, tra una conformazione e la sua immagine speculare si sceglie per convenzione quella in cui **il primo turno diverso da FORWARD è un LEFT**. Se è un RIGHT, si scambiano tutti i LEFT con i RIGHT. La catena tutta dritta non ha un'immagine speculare distinta ed è già canonica.

**4. La chiave.** I turni canonici sono una lista di cifre 0, 1, 2: lette da sinistra a destra come un **numero in base 3**, danno un unico numero intero, la chiave. Il primo turno è la cifra più significativa, l'ultimo quella delle unità.

*Esempio* ($N = 7$, quindi 5 turni). Turni `[FORWARD, RIGHT, RIGHT, LEFT, FORWARD]`, cifre `0 2 2 1 0`.

- Il primo turno non dritto è un RIGHT: la forma non è canonica, quindi si specchia e si ottiene `0 1 1 2 0`.

- I pesi delle 5 posizioni sono $3^4 = 81$, $3^3 = 27$, $3^2 = 9$, $3^1 = 3$, $3^0 = 1$:

  $$(0 \cdot 81) + (1 \cdot 27) + (1 \cdot 9) + (2 \cdot 3) + (0 \cdot 1) = 42$$

La chiave è 42. La forma non specchiata avrebbe dato 75: la chiave canonica è sempre la più piccola delle due, proprio perché la convenzione mette un 1 e non un 2 al primo cambio di direzione. Dalla chiave si torna ai turni dividendo ripetutamente per 3 e leggendo i resti al contrario ($42 \to 0,\ 14 \to 2,\ 4 \to 1,\ 1 \to 1,\ 0 \to 0$, cioè `0 1 1 2 0`).

La chiave è salvata come intero a 64 bit senza segno (`uint64`), che basta per catene fino a $N = 42$ residui ($3^{40} \approx 1{,}2 \times 10^{19} < 2^{64}$).

**In sintesi:** ogni forma fisica corrisponde a **un solo numero**, qualunque sia la sua posizione, il suo orientamento o il suo verso sulla griglia. Due conformazioni sono la stessa se e solo se hanno la stessa chiave, e confrontare due numeri interi è immediato.

#### 3.1.2 Come si costruisce l'elenco

**Passo 1: raccogliere tutte le chiavi.** Si mettono in un'unica lista le chiavi che vengono da tre fonti:

| Fonte (`source`) | Cosa contiene                                                  | Informazioni in più                          |
| ---------------- | -------------------------------------------------------------- | -------------------------------------------- |
| `snapshot`       | gli stati di tutti gli snapshot (§3)                           | replica, temperatura, ciclo, walker          |
| `trajectory`     | gli stati di tutte le traiettorie dei walker, mossa dopo mossa | walker, posizione nella traiettoria (`step`) |
| `remc_best`      | la **conformazione migliore** restituita dalla REMC            | nessuna                                      |

La terza fonte è una sicurezza. Le traiettorie si fermano dopo `max_recorded_steps` (50 000) mosse per walker: se un walker raggiunge lo stato a energia minima solo dopo, quello stato non sarebbe in nessuna traiettoria e forse in nessuno snapshot. La REMC però ricorda sempre lo stato migliore che ha visto, e così entra comunque nell'elenco.

**Passo 2: togliere i doppioni.** Nella lista la stessa chiave compare molte volte. Se ne tiene **una sola copia** (`np.unique`). Siccome la chiave è canonica, in questo modo si eliminano anche le copie ruotate, traslate e specchiate della stessa forma.

**Passo 3: decidere quali informazioni tenere.** Di uno stato che compare più volte si tengono le informazioni della **prima volta che compare nella lista**, cioè in quest'ordine:

1. prima gli snapshot, perché sono gli unici a conoscere la temperatura;
2. poi le traiettorie;
3. infine `remc_best`.

*Esempio.* Uno stato compare nello snapshot del ciclo 7 nella replica 2 (temperatura 0,38), e anche nella traiettoria del walker 3 alla mossa 1 200. Nell'elenco viene salvato con `source = snapshot`, `replica = 2`, `temperature = 0.38`, `cycle = 7` e `step = −1`, perché gli snapshot vengono prima delle traiettorie.

Di conseguenza `source` dice **da quale fonte è stato preso** lo stato, non **quando** è stato visitato per la prima volta: la traiettoria può averlo incontrato molto prima dello snapshot.

**Risultato.** Per una sequenza di 16 residui la REMC di default visita tra circa 8 000 e 103 800 stati distinti (mediana circa 25 900; colonna `visited_states` di `sequences.csv`). Per ognuno si calcola l'energia, e da questo insieme si scelgono nativi e decoy.

### 3.2 Perché una REMC più lunga e più fredda di quella del policy dataset

|                   | policy dataset | discriminator dataset |
| ----------------- | -------------- | --------------------- |
| `t_min` … `t_max` | 0,16 … 1,00    | 0,16 … 0,60           |
| `n_cycles`        | 200            | 500                   |

Qui serve soprattutto che la REMC **raggiunga lo stato fondamentale e ne trovi molte copie degeneri**: senza nativi non c'è la classe positiva, e una sequenza in cui la REMC non arriva al minimo viene scartata (§7). Le temperature quindi sono più basse e la simulazione più lunga. La replica più calda (0,60) basta comunque a visitare forme poco compatte, da cui vengono i decoy `far`.

---

## 4. I nativi (label 1)

### 4.1 Tutti i minimi, non solo il migliore

Come nel policy dataset, **lo stato nativo è l'insieme di tutte le conformazioni a energia minima**, non la singola "migliore" trovata dalla REMC. Nel modello HP l'energia minima si ottiene spesso con molte forme diverse: nel dataset di default il numero di nativi trovati va da 1 a 46 190 a seconda della sequenza (mediana 37).

Se si prendesse come positivo un solo stato e tutti gli altri come negativi, alcune forme a energia minima finirebbero etichettate come decoy: il modello imparerebbe che una conformazione ottima è "sbagliata". Per questo **qualsiasi stato con `energy == e_min` è un nativo**, e nessun decoy ha quell'energia.

### 4.2 Quanti se ne prendono

Si prendono tutti i nativi trovati, fino a `examples_per_class = 20`. Se sono di più se ne scelgono 20 a caso, senza ripetizioni (con un generatore casuale che dipende da `seed` e dal numero della sequenza, quindi riproducibile).

Il limite serve a non far pesare troppo le sequenze molto degeneri: senza, una sola sequenza con 10 000 nativi darebbe più esempi di tutte le altre messe insieme. Nel dataset di default 1 293 sequenze su 2 000 (il 65%) arrivano al limite di 20; in media una sequenza dà 15,7 nativi.

### 4.3 L'energia minima è davvero minima?

`e_min` è l'energia più bassa **tra gli stati visitati dalla REMC**, quindi in principio potrebbe non essere il vero minimo. Se così fosse, i "nativi" sarebbero in realtà decoy e il dataset sarebbe sbagliato alla radice. Per evitarlo, fino a `exact_check_max_length = 20` residui l'energia minima viene confrontata con quella dell'enumerazione esatta (`_core.exact_ground_state`, lo stesso codice della fase di verifica): se la REMC non l'ha raggiunta, la sequenza viene scartata (§7). Il valore esatto è salvato nella colonna `exact_e_min` di `sequences.csv`.

Oltre i 20 residui il controllo non si fa: `e_min` è il minimo della REMC e i nativi sono "i migliori stati trovati".

---

## 5. I decoy (label 0)

### 5.1 Quali stati possono essere decoy

Tutti gli stati visitati con **energia strettamente maggiore** di `e_min`. Per ogni sequenza se ne prendono **tanti quanti i nativi** (§6.1).

Prendere i decoy dalla REMC, invece che generarli a caso, è una scelta voluta: una conformazione casuale è quasi sempre srotolata e senza contatti, e distinguerla da un nativo è banale. Gli stati della REMC sono invece le forme che una ricerca fisica visita davvero, comprese quelle compatte e quasi native in cui si ferma più spesso. Sono i decoy che un discriminatore utile deve saper riconoscere.

### 5.2 Le fasce di difficoltà

Un decoy con gap 1 (un solo contatto in meno del nativo) è molto più difficile da riconoscere di uno con gap 6. Per avere esempi di ogni difficoltà, i decoy sono divisi in tre fasce secondo il **gap relativo**

$$
\text{frazione} = \frac{\text{gap}}{\lvert E_{min} \rvert}
$$

cioè la parte dei contatti del nativo che manca. Il denominatore $\lvert E_{min} \rvert$ è il gap più grande possibile, quello di uno stato senza contatti. Con `tier_edges = [0.34, 0.67]`:

| Fascia | Condizione                      | Significato                                     |
| ------ | ------------------------------- | ----------------------------------------------- |
| `near` | gap = 1, oppure frazione ≤ 0,34 | manca al più circa un terzo dei contatti        |
| `mid`  | 0,34 < frazione ≤ 0,67          | ne manca tra un terzo e due terzi               |
| `far`  | frazione > 0,67                 | ne mancano più di due terzi, forme quasi aperte |

Il gap 1, il più piccolo possibile, è sempre `near`, anche quando $\lvert E_{min} \rvert$ è piccolo.

*Esempio.* Con $E_{min} = -7$: gap 1 e 2 sono `near` (2/7 = 0,29), gap 3 e 4 sono `mid` (4/7 = 0,57), da gap 5 in su sono `far`. Con $E_{min} = -2$: gap 1 è `near`, gap 2 (energia 0) è `far`.

Usare il gap relativo e non quello assoluto rende le fasce confrontabili tra sequenze diverse: un gap di 2 è "quasi nativo" per una sequenza con $E_{min} = -8$, ma è una forma senza contatti per una con $E_{min} = -2$.

**Quote.** I decoy di una sequenza sono divisi **in parti uguali** tra le tre fasce (`tier_quotas`); il resto della divisione va alle fasce più vicine. Se una fascia non ha abbastanza stati, la parte che manca passa alle altre, in ordine `near` → `mid` → `far`. Per esempio, con 7 decoy da scegliere e abbastanza stati in ogni fascia le quote sono 3, 2, 2; se la fascia `near` ha un solo stato e la `far` nessuno, diventano 1, 6, 0.

Dentro ogni fascia gli stati sono scelti a caso.

### 5.3 Le trappole cinetiche

Una **trappola cinetica** è un decoy da cui **nessuna mossa singola abbassa l'energia**: tutte le conformazioni raggiungibili con una mossa (qualsiasi slot dello spazio delle azioni, VSHD o pull move, sul reticolo libero) hanno energia uguale o maggiore. È un minimo locale del paesaggio energetico: una discesa d'energia si ferma lì, e un Monte Carlo a bassa temperatura ci può restare a lungo. Per definizione nessun nativo è una trappola (il campo vale sempre 0 per i nativi).

Sono i decoy più insidiosi: compatti, stabili rispetto a piccoli cambiamenti, ma non nativi. Per questo nella fascia `near`:

1. si mescolano a caso i candidati della fascia;
2. si controllano i primi `trap_search_limit = 200` per vedere quali sono trappole (il controllo richiede di generare tutte le mosse dello stato, quindi non si fa su tutti);
3. si riserva alle trappole fino a `trap_fraction = 50%` della quota della fascia;
4. il resto della quota si riempie con gli altri candidati `near`, scelti a caso (tra questi possono capitare altre trappole).

Il campo `kinetic_trap` viene comunque calcolato per **tutti** i decoy scelti, anche `mid` e `far`. Nel dataset di default le trappole sono 10 810: 8 065 nella fascia `near` (il 67% della fascia), 2 366 nella `mid` e 379 nella `far`.

### 5.4 La sovrapposizione con il nativo

`native_overlap` dice quanto un decoy "assomiglia" a uno stato nativo, guardando i contatti H-H invece della forma:

$$
\text{overlap} = \max_{\text{nativi } k} \frac{\#\{\text{contatti H-H del nativo } k \text{ presenti anche nel decoy}\}}{\#\{\text{contatti H-H del nativo } k\}}
$$

Il massimo è preso sui nativi **scelti come esempi** per quella sequenza. Vale 1 per un nativo; per un decoy è vicino a 1 se ha quasi gli stessi contatti di un nativo (magari con una parte della catena piegata male), vicino a 0 se i suoi contatti sono altri.

È un'informazione diversa dal gap: due decoy con lo stesso gap possono essere "un nativo con un difetto" (overlap alto) oppure "un'altra forma compatta" (overlap basso). Nel dataset di default l'overlap medio è 0,60 nella fascia `near`, 0,39 nella `mid` e 0,12 nella `far`.

---

## 6. Bilanciamento e duplicati

### 6.1 Un decoy per ogni nativo, sequenza per sequenza

Ogni sequenza dà **lo stesso numero di nativi e di decoy** (bilanciamento 1:1 per sequenza, verificato dal test `test_balance_and_no_duplicates`). Così:

- le due classi sono bilanciate nel dataset intero, e un classificatore che risponde sempre "decoy" ha accuratezza 50%, non 95%;
- soprattutto, sono bilanciate **dentro ogni sequenza**. Se una sequenza desse molti più nativi che decoy, il modello potrebbe imparare a riconoscere la sequenza ("questa è una di quelle con tanti nativi") invece di guardare la conformazione.

Il prezzo è che le sequenze con pochi nativi danno pochi esempi: una sequenza con un solo stato fondamentale contribuisce con 2 esempi in tutto.

### 6.2 Niente duplicati

Ogni esempio è identificato dalla sua chiave canonica, e gli stati visitati sono già deduplicati (§3.1). Quindi **dentro una sequenza la stessa forma compare al più una volta**, e una forma non può essere sia nativa sia decoy. Le copie ruotate, traslate o specchiate della stessa forma contano come la stessa forma.

La stessa conformazione può invece comparire in sequenze diverse, anche con label diverse. Non è un problema: l'input del modello è la coppia (sequenza, conformazione).

---

## 7. Come vengono generate le sequenze

- Sono tutte **casuali**, di lunghezza `length` (16), con probabilità `h_fraction` (0,5) che ogni residuo sia H. A differenza del policy dataset, la sequenza di `protein.sequence` non viene inclusa.
- **Una sequenza e la sua inversa** descrivono la stessa catena letta dall'altro estremo: se ne tiene una sola (funzione `candidate_sequences`, condivisa con il policy dataset). Altrimenti lo stesso problema potrebbe finire sia nel train sia nel test.
- Una sequenza viene **scartata** se:
  - l'energia minima è **0** (nessun contatto H-H possibile): ogni stato sarebbe nativo e non esisterebbero decoy;
  - la REMC **non ha raggiunto lo stato fondamentale** esatto (solo fino a `exact_check_max_length` residui, §4.3): i suoi "nativi" sarebbero decoy.
- Si generano sequenze finché non se ne hanno `n_sequences` (2000) valide. Se se ne scartano più di 20 volte tante, la generazione si ferma con un errore.

Le sequenze scartate non consumano un numero: `seq_id` conta solo quelle accettate, e la REMC della sequenza `i` usa il seme `remc.seed + i`. Il numero di sequenze scartate per ogni motivo è scritto in `report.txt` e in `meta.json`. Nel dataset di default ne sono state scartate 13, tutte per energia minima 0; nessuna per stato fondamentale mancato.

Le energie minime delle 2000 sequenze di default vanno da −1 a −9:

```
E_min   −1   −2   −3   −4   −5   −6   −7   −8   −9
seq.     9  124  139  394  448  474  287  121    4
```

---

## 8. Divisione in train, validation e test

Le 2000 sequenze vengono divise a caso in tre gruppi:

| Insieme      | Sequenze | A cosa serve                                                 |
| ------------ | -------- | ------------------------------------------------------------ |
| `train`      | 1600     | addestrare il modello                                        |
| `val`        | 200      | scegliere i parametri del modello durante l'addestramento   |
| `test`       | 200      | misurare alla fine quanto il modello funziona               |

Le proporzioni sono quelle di `split = [0.8, 0.1, 0.1]`, e la divisione la fa la funzione `assign_splits`, la stessa del policy dataset.

**Si divide per sequenza, non per esempio:** tutti gli esempi di una sequenza, cioè i suoi nativi e i suoi decoy, finiscono nello stesso insieme. In questo modo le sequenze del test sono sequenze che il modello **non ha mai visto**.

*Perché.* I nativi di una sequenza si assomigliano molto: spesso differiscono solo per pochi residui. Se una parte finisse in train e un'altra in test, il modello riconoscerebbe i nativi del test perché ne ha già visti di quasi identici, non perché ha imparato a distinguere un nativo da un decoy. Il risultato sul test sembrerebbe migliore di quello che è.

Siccome ogni sequenza ha tanti nativi quanti decoy, anche ogni insieme è bilanciato 1:1.

---

## 9. I file

### 9.1 Elenco

| File                               | Contenuto                                                                                                                                                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `train.csv`, `val.csv`, `test.csv` | Gli esempi di ogni split, una riga per esempio (§9.2).                                                                                                                                                                                           |
| `sequences.csv`                    | Una riga per sequenza (§9.3).                                                                                                                                                                                                                    |
| `meta.json`                        | Descrizione del dataset: versione del formato, lunghezza N, nomi degli split, delle fasce e delle sorgenti, significato delle lettere dei turni, convenzione delle coordinate, il riepilogo (`summary`), la configurazione usata, il commit git. |
| `report.txt`                       | Il riepilogo stampato a fine esecuzione: esempi per classe e split, sequenze per split, sequenze scartate, fasce, trappole, istogramma dei gap, energie dei decoy.                                                                               |

Il formato è CSV, leggibile anche con un foglio di calcolo o con pandas. Il dataset è piccolo (circa 62 600 esempi, 12,9 MB): non serve la compressione del policy dataset.

### 9.2 Campi di un esempio

| Campo            | Contenuto                                                                                           |
| ---------------- | --------------------------------------------------------------------------------------------------- |
| `seq_id`         | numero della sequenza in `sequences.csv`                                                            |
| `sequence`       | la sequenza HP                                                                                      |
| `split`          | `train`, `val` o `test`                                                                             |
| `label`          | **1** nativo, **0** decoy                                                                           |
| `energy`         | energia HP della conformazione                                                                      |
| `e_min`          | energia minima della sequenza                                                                       |
| `gap`            | `energy − e_min` (0 per i nativi)                                                                   |
| `tier`           | `native`, `near`, `mid` o `far` (§5.2)                                                              |
| `kinetic_trap`   | 1 se il decoy è un minimo locale dell'energia (§5.3); sempre 0 per i nativi                         |
| `native_overlap` | frazione dei contatti H-H di un nativo presenti nello stato, la più alta tra i nativi scelti (§5.4) |
| `turns`          | turni della forma canonica, una lettera per residuo interno: `F`, `L`, `R`                          |
| `coordinates`    | coordinate della forma canonica come lista JSON `[[x0,y0],[x1,y1],…]`                               |
| `key`            | chiave canonica                                                                                     |
| `source`         | dove è stato trovato lo stato: `snapshot`, `trajectory` o `remc_best` (§3.1)                        |
| `walker`         | walker dello snapshot o della traiettoria (−1 per `remc_best`)                                      |
| `replica`        | replica dello snapshot (−1 altrimenti)                                                              |
| `temperature`    | temperatura della replica dello snapshot (vuoto altrimenti)                                         |
| `cycle`          | ciclo REMC dello snapshot (−1 altrimenti)                                                           |
| `step`           | posizione nella traiettoria del walker (−1 altrimenti)                                              |

Nel dataset di default i nativi vengono quasi tutti dagli snapshot (26 005 su 31 320), soprattutto dalle repliche fredde, mentre i decoy vengono quasi tutti dalle traiettorie (29 199 su 31 320). È una conseguenza dell'ordine di §3.1: uno stato a energia minima viene visto spesso, quindi è facile che compaia anche in uno snapshot; un decoy qualsiasi è uno tra decine di migliaia di stati di passaggio, che quasi sempre compaiono solo nelle traiettorie. Anche per questo i campi di provenienza non vanno usati come feature.

### 9.3 Campi di `sequences.csv`

| Campo                         | Contenuto                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------ |
| `seq_id`, `sequence`, `split` | numero, sequenza, split                                                        |
| `e_min`                       | energia minima tra gli stati visitati                                          |
| `exact_e_min`                 | energia minima dell'enumerazione esatta (vuoto oltre `exact_check_max_length`) |
| `remc_best_energy`            | energia della conformazione migliore restituita dalla REMC                     |
| `natives_found`               | stati distinti a energia minima trovati dalla REMC (prima del limite)          |
| `positives`                   | nativi presi come esempi (al più `examples_per_class`)                         |
| `decoys`                      | decoy presi come esempi (uguale a `positives`)                                 |
| `visited_states`              | stati distinti visitati dalla REMC                                             |

---

## 10. Come si usa

### 10.1 Lettura

```python
from hpfold import discriminator_dataset

data = discriminator_dataset.load_split("data/output/datasets/discriminator_dataset", "train")
meta = discriminator_dataset.load_meta("data/output/datasets/discriminator_dataset")

X_seq = data["hp"]            # (S, N): 1 per H, 0 per P
X_turns = data["turns"]       # (S, N−2): 0 = FORWARD, 1 = LEFT, 2 = RIGHT
X_coords = data["coordinates"]  # (S, N, 2)
y = data["label"]             # (S,): 1 nativo, 0 decoy
```

`load_split` restituisce un dizionario di array numpy: le colonne del CSV (numeri e stringhe) più tre codifiche già pronte, `hp`, `turns` come codici numerici e `coordinates` come array. Da qui si ottengono le altre codifiche con le funzioni del policy dataset, per esempio:

```python
from hpfold import policy_dataset

contacts = policy_dataset.contact_map(data["coordinates"])            # (S, N, N)
grid = policy_dataset.lattice_grid(data["coordinates"], data["hp"])   # (S, 3, N, N)
```

### 10.2 Target

- **Classificazione:** `data["label"]`.
- **Regressione o ranking:** `data["gap"]` (o `data["energy"]`). Un modello che impara a ordinare le forme per gap può poi fare da classificatore con una soglia.
- **Pesi:** `tier` e `kinetic_trap` permettono di dare più peso ai decoy difficili, oppure di valutare il modello fascia per fascia.

### 10.3 Simmetrie

Come nel policy dataset, ogni stato è salvato **una sola volta in forma canonica**, quindi il primo turno non dritto è sempre `L`. Un modello allenato solo su questi dati vedrebbe solo catene che iniziano a piegarsi a sinistra. Conviene quindi specchiare a caso una parte degli esempi al caricamento:

- **specchio:** si scambiano `L` e `R` nei turni (codici 1 ↔ 2), oppure si cambia segno alla coordinata $y$;
- **rotazione:** serve solo alle codifiche a coordinate o a griglia; `policy_dataset.rotate_coordinates` ruota di 90°.

A differenza del policy dataset non ci sono slot di mosse da permutare: label, energia, gap e overlap non cambiano con una rotazione o una riflessione, quindi l'augmentation tocca solo l'input.

---

## 11. Parametri (`config/default.toml`)

### `[discriminator_dataset]`

| Parametro                | Default         | Significato                                                                      |
| ------------------------ | --------------- | -------------------------------------------------------------------------------- |
| `n_sequences`            | 2000            | sequenze valide nel dataset (con energia minima < 0)                             |
| `length`                 | 16              | residui di ogni sequenza (tra 4 e la lunghezza massima della chiave canonica)    |
| `h_fraction`             | 0.5             | probabilità che un residuo sia H                                                 |
| `seed`                   | 11              | seme per sequenze, split e scelta degli esempi                                   |
| `split`                  | [0.8, 0.1, 0.1] | proporzioni di train, validation e test, per sequenza                            |
| `examples_per_class`     | 20              | nativi al massimo per sequenza; i decoy sono altrettanti                         |
| `tier_edges`             | [0.34, 0.67]    | soglie del gap relativo tra `near`/`mid` e tra `mid`/`far`                       |
| `trap_fraction`          | 0.5             | parte della quota `near` riservata alle trappole cinetiche                       |
| `trap_search_limit`      | 200             | candidati `near` controllati per trovare le trappole                             |
| `exact_check_max_length` | 20              | fino a questa lunghezza l'energia minima è confrontata con l'enumerazione esatta |
| `max_recorded_steps`     | 50000           | mosse registrate al massimo per ogni walker                                      |

### `[discriminator_dataset.remc]`

La REMC lanciata per ogni sequenza: stesse chiavi di `[remc]` (`n_replicas` = 5, `t_min` = 0,16, `t_max` = 0,60, `local_steps` = 100, `n_cycles` = 500, `rho` = 0,4, `seed` = 1000), più `save_every` = 2, cioè ogni quanti cicli salvare uno snapshot. La sequenza i usa il seme `seed + i`. È indipendente da `[policy_dataset.remc]`.

---

## 12. Il dataset di default in numeri

|                                   |                                                                  |
| --------------------------------- | ---------------------------------------------------------------- |
| sequenze                          | 2000 di 16 residui (1600 train, 200 validation, 200 test); 13 scartate |
| energia minima                    | da −1 a −9, sempre uguale a quella esatta                              |
| stati visitati dalla REMC         | da circa 8 000 a 103 800 per sequenza (mediana 25 900)                 |
| nativi trovati per sequenza       | da 1 a 46 190 (mediana 37); 1 293 sequenze arrivano al limite di 20    |
| esempi                            | 62 640: 31 320 nativi e 31 320 decoy                                   |
| esempi per split (nativi / decoy) | train 25 177 / 25 177, val 3 028 / 3 028, test 3 115 / 3 115           |
| decoy per fascia                  | 12 095 `near`, 9 953 `mid`, 9 272 `far`                                |
| trappole cinetiche                | 10 810 (8 065 nella fascia `near`)                                     |
| gap dei decoy                     | da 1 a 8; il 55% ha gap 1 o 2                                          |
| `native_overlap` medio            | 0,60 `near`, 0,39 `mid`, 0,12 `far`                                    |
| spazio su disco                   | circa 12,9 MB                                                          |
| tempo di generazione              | circa 31 minuti (circa 0,9 s per sequenza)                             |

---

## 13. Differenze rispetto al policy dataset

|                           | Policy dataset                                                    | Discriminator dataset                                     |
| ------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- |
| Domanda per il modello    | "quale mossa faccio adesso?"                                      | "questa forma è nativa?"                                  |
| Target                    | distanza dopo ogni mossa (107 slot)                               | label 0/1, più energia e gap                              |
| Chi dà le etichette       | l'oracolo (BFS sul grafo delle mosse)                             | l'energia: nativo ⇔ energia minima                        |
| Stati usati               | snapshot, percorsi ottimi, traiettorie ripulite, stati perturbati | stati distinti visitati dalla REMC                        |
| Bilanciamento             | nessuno                                                           | 1:1 nativi/decoy per sequenza                             |
| Esempi                    | ~75 600 record                                                    | ~62 600 esempi                                            |
| Formato                   | `.npz` (numpy)                                                    | `.csv`                                                    |
| Sezioni di configurazione | `[policy_dataset]`, `[policy_dataset.remc]`                       | `[discriminator_dataset]`, `[discriminator_dataset.remc]` |

In comune: la REMC con `SampleRecorder`, la forma canonica e la chiave, la generazione delle sequenze (senza inverse doppie), lo split per sequenza e l'idea che lo stato nativo sia l'insieme di tutti i minimi.

---

## 14. Dove sta il codice

| File                                                                    | Contenuto                                                                                                                          |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [discriminator_dataset.py](../python/hpfold/discriminator_dataset.py)   | tutta la pipeline: energie, trappole, fasce, scelta degli esempi, file, report, lettura                                            |
| [policy_dataset.py](../python/hpfold/policy_dataset.py)                 | funzioni riusate: `candidate_sequences`, `assign_splits`, `keys_to_turns`, `turns_to_coordinates`, `contact_map`, `straight_chain` |
| [sample_writer.hpp](../cpp/include/hp/sample_writer.hpp)                | `SampleRecorder`: registrazione delle traiettorie e degli snapshot REMC                                                            |
| [state_graph.hpp](../cpp/include/hp/state_graph.hpp)                    | `successors`: le conformazioni raggiungibili con una mossa, usate per le trappole cinetiche                                        |
| [test_discriminator_dataset.py](../tests/test_discriminator_dataset.py) | i test (§15)                                                                                                                       |

Le funzioni principali di `discriminator_dataset.py`:

- `sample_remc`: lancia la REMC di una sequenza con il `SampleRecorder`;
- `visited_states`: gli stati distinti visitati, con la loro provenienza (§3.1);
- `energies`: l'energia HP di una lista di chiavi, calcolata in numpy dalle matrici di contatto;
- `is_kinetic_trap`, `tier_of`, `tier_quotas`, `native_overlap`: le misure di §5;
- `sequence_examples`: nativi e decoy di una sequenza, oppure il motivo per cui è scartata;
- `build_dataset`, `run`: generazione dei file e report;
- `load_split`, `load_meta`: lettura.

---

## 15. Cosa verificano i test

`tests/test_discriminator_dataset.py` controlla:

- **energie:** l'energia calcolata in numpy coincide con quella del C++ lungo un cammino casuale di 300 mosse;
- **trappole cinetiche:** la catena dritta non è una trappola, lo stato fondamentale esatto sì;
- **fasce e quote:** `tier_of` e `tier_quotas` su casi noti, compreso quello con decoy insufficienti;
- **file generati** (su un dataset piccolo, 6 sequenze di 10 residui):
  - ci sono tutti i file e le colonne sono quelle previste;
  - per ogni esempio, ricostruendo la conformazione dalle coordinate si ritrovano energia e chiave; il gap è giusto; i nativi hanno energia minima e i decoy energia maggiore;
  - ogni sequenza ha tanti nativi quanti decoy, al più `examples_per_class`, e nessuna chiave ripetuta; l'energia minima coincide con quella esatta;
  - gli split non hanno sequenze in comune e non contengono una sequenza insieme alla sua inversa;
  - `load_split` restituisce array delle forme giuste;
  - rigenerando il dataset con la stessa configurazione si ottengono file identici;
- **riga di comando:** la fase `discriminator_dataset` esiste ed è separata da `policy_dataset`, e le due cartelle di output sono diverse.

---

## 16. Limiti e cose non fatte

- **Pochi esempi per sequenza.** Circa 62 600 esempi, limitati da `examples_per_class` e dal bilanciamento 1:1: ogni sequenza ne dà al più 40. Si possono aumentare con più sequenze o alzando il limite.
- **Decoy solo dalla REMC.** I decoy sono stati visitati dalla REMC di quella sequenza: forme che la REMC non visita (per esempio quelle di un altro metodo, come una policy che sbaglia) non sono rappresentate. In seguito si potranno aggiungere i decoy prodotti dal modello di folding.
- **Oltre 20 residui i nativi non sono garantiti.** Senza l'enumerazione esatta, `e_min` è solo il minimo trovato dalla REMC.
- **Sequenze molto degeneri tagliate.** Per le sequenze con migliaia di nativi se ne vedono solo 20, scelti a caso.
- **Una lunghezza per dataset.** Tutte le sequenze hanno `length` residui. Per il discriminatore non è un vincolo strutturale come per il policy dataset (non ci sono slot), ma la pipeline genera comunque una sola lunghezza.
- **I campi di provenienza sono sbilanciati tra le classi** (§9.2): vanno usati solo per l'analisi, mai come feature.
