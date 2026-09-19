# Pull moves

Sono state progettate per risolvere il problema del **blocco sterico** (*topological jamming*).

Nelle conformazioni molto dense (compatte), i set di mosse tradizionali come i *kink-flip* o le *end-moves* falliscono perché richiedono che le celle adiacenti a un singolo residuo siano vuote. Le *pivot moves*, d'altra parte, tentano di ruotare interi blocchi di catena, scontrandosi quasi sempre con altri residui occupati.

L'idea fondamentale delle Pull Moves è fare **scivolare la catena lungo il proprio percorso**: invece di spingere i residui verso celle esterne vuote, una pull move "stira" un residuo verso una posizione adiacente e fa scorrere i residui precedenti occupando le posizioni appena liberate dalla catena stessa.

## **Lesh, Mitzenmacher e Whitesides (RECOMB 2003)**

La particolarità dell'algoritmo di Lesh et al. sta nella sua **sicurezza sterica**:

- L'unico nodo del reticolo per cui devi verificare preventivamente che sia libero è il **sito target iniziale $L$**.

- Tutti gli altri residui coinvolti nella mossa si spostano **occupando la posizione appena lasciata libera dal residuo precedente**. Poiché quella posizione era occupata un istante prima, è garantito al 100% che sia priva di altri atomi.

### La Meccanica Passo-Passo (Reticolo 2D)

Consideriamo una catena su un reticolo quadrato composto da:

- Residuo $i$

- Residuo adiacente $i+1$

- L: nodo vuoto

- C: Posizione adiacente a $i$  in cui può essere presente o meno un residuo. L'algoritmo si comporta di conseguenza

 Selezioniamo due residui adiacenti lungo la sequenza: $i$ e $i+1$.

```
Stato Iniziale (Esempio):

   (C)     (L)      <- L è un nodo VUOTO, C è una posizione adiacente a i
    |       
   (i) ─── (i+1)
```

#### 1. Condizioni di Innesco

Per poter avviare una pull move sul residuo $i$ verso il residuo $i+1$:

- Deve esistere un nodo **vuoto $L$** adiacente a $i+1$.

- $L$ deve essere **diagonale** rispetto a $i$

#### 2. La Mossa Iniziale

Il residuo $i$ viene spostato dal suo nodo attuale $P_i$ al nodo target $L$.

```
Dopo lo spostamento di i in L:

   (C) ─── (i)      <- i si è mosso in L!
            |       
   (P_i)   (i+1)    <- P_i è la vecchia posizione di i, ora VUOTA
```

#### 3. La Propagazione (Il "Dominò" Reversibile)

A questo punto la catena potrebbe essersi "spezzata" geometricamente tra $i$ e $i-1$. Si controlla la posizione del residuo $i-1$:

- **Caso A (Arresto Immediato):** Se $i-1$ si trovava già nel nodo $C$ (adiacente sia a $i$ nella sua nuova posizione $L$, sia a $P_i$), la connettività della catena è già garantita. **La mossa termina qui.**

- **Caso B (Propagazione):** Se $i-1$ **non** è adiacente a $L$, allora $i-1$ viene spostato ad occupare la posizione $P_i$ (appena liberata da $i$).

  - Successivamente si controlla $i-2$: se è adiacente a $P_i$, ci si ferma. Altrimenti, $i-2$ si sposta nella vecchia posizione di $i-1$, e così via.

La mossa si propaga lungo la catena (verso il residuo $1$ o simmetricamente verso $N$) finché **non si ripristina la continuità topologica** o non si raggiunge l'estremità della catena.

### Adiacenza spezzata tra due residui

Sul reticolo quadrato due punti sono adiacenti (connessi da un legame) **solo se si trovano a distanza $1$** (cioè se condividono un lato della griglia, con distanza Manhattan $\Delta x + \Delta y = 1$).

# Alternative alle pull moves

Di seguito sono riportati alcuni lavorati che forniscono una variazione alle pull moves solo per quanto riguarda la scelta geometriche delle mosse. Non viene trattato alternative al campionamento per l'algoritmo Metropolis

# Pivot, Crank-shaft and End moves

### 1. Pivot Moves (Mosse Pivot)

La **Pivot Move** è una mossa **globale** progettata per ristrutturare ampie porzioni della catena in un singolo step.

```
Stato Iniziale:        [1]──[2]──[3]──[4]──[5]──[6]  (Perno su [3])
                                   │
                                 Rotazione di 90° a valle
                                   │
Nuova Proposta:        [1]──[2]──[3]
                                   │
                                 [4]──[5]──[6]
```

- **Come funziona:**

  1. Si sceglie casualmente un residuo interno $i$ lungo la catena, che agisce da "perno" (*pivot*).

  2. La catena si divide ideale in due parti: la porzione $1 \dots i$ rimane completamente fissa nello spazio.

  3. Si seleziona una trasformazione di simmetria ammessa dal reticolo (es. su reticolo 2D: rotazione di 90°, 180°, 270° o una riflessione speculare).

  4. La trasformazione viene applicata **a tutta la sottocatena rimanente** ($i+1 \dots N$), facendola ruotare o riflettere in blocco attorno al perno $i$.

- **Vantaggi:** Consente alla simulazione di compiere enormi "salti" nello spazio delle configurazioni, modificando drasticamente l'orientamento globale della proteina in un unico passo.

- **Svantaggi e Limiti:** È estremamente efficace per catene srotolate (ad alte temperature). Tuttavia, quando la proteina si compatta verso lo stato nativo, ruotare metà catena provoca quasi sempre collisioni steriche con l'altra metà fissa. Nello stato compatto, il tasso di accettazione delle Pivot Moves **crolla verso lo 0%**.

### 2. Crank-shaft Moves (Mosse ad Albero a Gomiti)

La **Crank-shaft Move** è una mossa **locale** che agisce solo su una piccola sezione interna della proteina, mantenendo inalterato tutto il resto della struttura.

```
Stato Iniziale:        [i]──[i+1]            Asse di rotazione tra [i] e [i+3]
                        │     │              I residui [i+1] e [i+2] ruotano
                       [i+3]─[i+2]           dall'altro lato dell'asse.
                        
Nuova Proposta:        [i]───[i+3]
                        │     │
                       [i+1]─[i+2]
```

- **Come funziona:**

  1. Si individuano 4 residui consecutivi lungo la sequenza $(i, i+1, i+2, i+3)$ che formino una struttura piegata a "U" (tre lati di un quadrato sul reticolo).

  2. I residui estremi $i$ e $i+3$, così come l'intera catena al di fuori di questo blocco, rimangono rigidi e immobili.

  3. I due residui interni ($i+1$ e $i+2$) vengono ruotati attorno all'asse virtuale formato da $i$ e $i+3$:

     - **In 2D:** Significa "ribaltare" la piega a "U" dal lato opposto dell'asse (se i nodi d'arrivo sono liberi).

     - **In 3D:** Significa ruotare i due residui attorno all'asse di 90°, 180° o 270°.

- **Vantaggi:** Poiché muove solo due monomeri e lascia intatti i terminali e il resto della catena, la perturbazione energetica è limitata e locale.

- **Svantaggi e Limiti:** È applicabile **solo** quando la geometria locale presenta una configurazione a "U". Inoltre, non è in grado di modificare la forma globale della catena, ma serve principalmente a riorganizzare i contatti locali.

### 3. End Moves (Mosse Terminali)

La **End Move** è la mossa locale più semplice in assoluto e si applica esclusivamente ai due estremi della sequenza.

```
Stato Iniziale:        [1]──[2]──[3]──[4]...  (Spostamento del terminale [1])
                        
Nuova Proposta:        [1]
                        │
                       [2]──[3]──[4]...
```

- **Come funziona:**

  1. Si seleziona uno dei due residui terminali della catena: il primo ($i=1$) o l'ultimo ($i=N$).

  2. Si individua la posizione del suo vicino sequenziale ($i=2$ o $i=N-1$).

  3. Si sposta il residuo terminale in una qualsiasi posizione adiacente a $i=2$ (o $N-1$) sul reticolo che sia attualmente vuota, ruotando il solo legame finale.

- **Vantaggi:** Ha un **tasso di accettazione altissimo** in qualsiasi stadio della simulazione. Poiché le estremità della proteina si trovano spesso verso l'esterno del *core* idrofobico, incontrano pochissimi ostacoli sterici.

- **Svantaggi e Limiti:** Agisce unicamente sui 2 monomeri agli estremi della proteina. Non può aiutare a risolvere i blocchi sterici o le riorganizzazioni all'interno del *core* compatto della struttura.

### Confronto Sintetico

| **Mossa**       | **Tipo** | **Monomeri Spostati**              | **Tasso di Accettazione (Stato Srotolato)** | **Tasso di Accettazione (Stato Compatto)** |
| --------------- | -------- | ---------------------------------- | ------------------------------------------- | ------------------------------------------ |
| **Pivot**       | Globale  | Intera sottocatena ($i+1 \dots N$) | **Molto Alto** (~30-80%)                    | **Quasi Nullo** (<1%)                      |
| **Crank-shaft** | Locale   | 2 monomeri interni ($i+1, i+2$)    | Medio                                       | **Buono** (se lo spazio attorno è libero)  |
| **End Move**    | Locale   | 1 monomero terminale ($1$ o $N$)   | **Altissimo**                               | **Molto Alto**                             |

# Move Set Ibridi e Compositi

L'utilizzo esclusivo di *pull moves* può generare inefficienze quando la catena è quasi completamente srotolata (dove le *pivot moves* sono drasticamente più veloci) o quando è necessario riordinare interi domini idrofobici senza disfare il "packing" locale.

```
[Stato Srotolato]  ──►  Pivot Moves        (Esplorazione globale rapida)
[Stato Intermedio] ──►  Pull Moves         (Compattazione ergodica)
[Stato Compatto]   ──►  Crank-shaft / End  (Raffinamento locale dei contatti)
```

- **Pivot + Pull Hybrid Moves (Shmygelska & Hoos, 2007):** Dimostrano che l'approccio migliore consiste nell'alternare le mosse in modo stocastico o dinamico in base al raggio di girazione della proteina:

  - Ad alte temperature / forme estese: 80% *Pivot*, 20% *Pull*.

  - A basse temperature / forme compatte: 90% *Pull*, 10% *Kink-flip/Crank-shaft*.

- **Multi-Point / Block Pull Moves:** Varianti in cui si effettuano due o più *pull moves* simultanee su punti diversi della catena all'interno dello stesso step Monte Carlo. Questo consente di superare barriere energetiche locali che una singola *pull move* non riuscirebbe a scalzare.

# Altre algoritmi per la ricerca del minimo locale

## Integrare le Pull Moves in Metaeuristiche (I Record di Convergenza)

I risultati numericamente migliori in assoluto nell'individuazione dello stato nativo per sequenze HP molto lunghe ($N \ge 100$) non sono stati ottenuti da simulazioni Monte Carlo pure, ma da algoritmi ibridi in cui le **pull moves sono usate come operatore di ricerca locale**.

#### A. ACO + Pull Moves (Ant Colony Optimization)

- **Shmygelska et al. (2005) / Hu et al. (2008):** Combinano l'algoritmo delle colonie di formiche con le *pull moves*.

- Le "formiche" costruiscono conformazioni candidate (*chain growth*), e successivamente una fase di ricerca locale basata su *pull moves* "strizza" e compatta le strutture trovando minimi energetici che nessun MCMC puro riesce a raggiungere in tempi comparabili.

#### B. Genetic Algorithms / Tabu Search + Pull Moves

- **Zhang et al. (2013) / Jiang et al. (2018):** Nei classici algoritmi genetici su reticolo, la fase di *crossover* (taglio e unione di due conformazioni) genera quasi sempre sovrapposizioni steriche (SAW non valida).

- **Soluzione:** Usare le *pull moves* come operatore di **mutazione/riparazione**: quando un crossover genera un'incompatibilità sterica, una serie di *pull moves* viene applicata per "riparare" la catena mantenendo l'impacchettamento dei core idrofobici.

## Estensione a Reticoli a Maggiore Coordinazione (FCC Lattice)

Un grande limite teorico del reticolo quadrato/cubico semplice (SC) è la **restrizione di parità**: due residui separati da un numero pari di monomeri lungo la sequenza non potranno mai trovarsi adiacenti nello spazio.

- **FCCI (Face-Centered Cubic) Pull Moves:** Adattare le *pull moves* al reticolo cubico a facce centrate (FCC, con numero di coordinazione $z=12$ invece di $z=6$).

- **Risultato:** Su reticolo FCC, il tasso di accettazione delle *pull moves* sale drasticamente, la barriera di *jamming* sterico si riduce e la simulazione approssima molto meglio la flessibilità dello spazio continuo.

# Scelta per il progetto

Per il progetto ho scelto le semplici pull moves da abbinare all'algoritmo di campionamento REMC (vedi file samplings.md)

## Evita complessità inutili

L'aggiunta di metaeuristiche avanzate (come Ant Colony Optimization o Algoritmi Genetici) o di reticoli complessi (come l'FCC) inserisce una complessità "inutile" se rapportata all'**obiettivo globale del tuo progetto d'esame**, che si articola in due fasi (Campionamento $\rightarrow$ Machine Learning).

Quattro ragioni spiegano perché questi approcci complessi rischiano di penalizzare il progetto anziché valorizzarlo:

#### 1. Divergenza di Scopo: Ottimizzazione vs. Campionamento Fisico

- **Cosa fanno le metaeuristiche (ACO, GA):** Sono algoritmi pensati quasi esclusivamente per la **ricerca del minimo globale** ($E_\text{min}$). Il loro obiettivo è trovare la struttura nativa nel minor tempo possibile, tagliando ed eliminando aggressivamente le strutture sub-ottimali o srotolate.

- **Cosa serve a te (Fase 1):** A te **non** serve solo trovare lo stato nativo, ma generare un **dataset bilanciato e diversificato** che contenga:

  - Conformazioni native ($E_\text{min}$).

  - Conformazioni *near-native* (quasi corrette, con piccoli errori).

  - Conformazioni *decoy* / srotolate (ad alta energia).

- **Il problema:** Un algoritmo come l'ACO o un GA distorcerebbe il dataset verso sole strutture compattissime, rendendo difficile estrarre la distribuzione di Boltzmann fisica necessaria per addestrare un modello di ML robusto. L'algoritmo **Metropolis / REMC**, invece, garantisce per costruzione teorica un campionamento termodinamico corretto a varie temperature.

#### 2. Impatto sulla Fase 2: Complessità delle Feature per il Machine Learning

- **Reticolo Quadrato/Cubico (2D/3D):** È facilmente rappresentabile in Python come matrice di adiacenza, griglia discreta (ideale per Reti Neurali Convoluzionali 2D/3D) o vettori di mosse cartesiane ($L, R, F, U, D$).

- **Reticolo FCC (Face-Centered Cubic):** Ha 12 vicini per ogni nodo invece di 4 o 6. Gestire la geometria di un reticolo FCC in codice aumenta la complessità matematica del *feature engineering* per il modello di Machine Learning senza aggiungere alcun valore concettuale alla tesi/progetto d'esame.

#### 3. Bilancio di Sforzo nell'Economia dell'Esame

Un progetto universitario viene valutato sulla **completezza e la coerenza dell'intero pipeline**:

$$
\text{Valore del Progetto} = \text{Qualità MCMC (Fase 1)} + \text{Qualità ML (Fase 2)} + \text{Analisi Risultati}
$$

Se dedichi l'80% del tempo di sviluppo a implementare una metaeuristica complessa come ACO + Pull Moves su reticolo FCC per la Fase 1:

- Rischi di arrivare con la Fase 2 (ML) trattata in modo superficiale o affrettato.

- Aumenti drasticamente le probabilità di bug nel codice di campionamento, rendendo difficile il debugging.

#### 4. *Pull Moves + REMC* sono già sufficienti per la scala del progetto

Le metaeuristiche estreme (ACO, GA) sono state inventate per risolvere sequenze lunghissime ($N \ge 100$ residui) dove lo spazio conformazionale è $3^{100} \approx 5 \times 10^{47}$ stati.

Per un progetto universitario, lavorare con sequenze HP di lunghezza $N \in [20, 50]$ è ideale:

- Lo spazio delle fasi è abbastanza grande da essere non banale.

- **Pull Moves + REMC** risolvono il blocco sterico in pochi secondi di calcolo su una CPU standard.

- Ottieni un dataset di migliaia di strutture perfettamente etichettate senza dover complicare l'architettura.
