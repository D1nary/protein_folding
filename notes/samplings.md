ricorda analisi Swetnam & Allen (2009)

# Sampling methods (con le pull moves)

Le **Pull Moves** offrono un set di mosse locali reversibile, ergodico e completo per modelli HP su reticolo, garantendo che la catena peptidica rimanga continua ed escluda auto-intersezioni ad ogni passaggio. Combinando le Pull Moves con l'algoritmo Metropolis-Hastings, le opzioni di campionamento dello spazio conformazionale si dividono diverse strategie principali, ciascuna con precisi vantaggi per la creazione del dataset di Machine Learning.

## Strategie di Campionamento Disponibili

### 1. Metropolis Canonico a Temperatura Fissa

Esegue la catena di Markov mantenendo una temperatura costante $T$. La probabilità di accettazione della mossa proposta è regolata dal fattore di Boltzmann:

$$
P_{\text{acc}} = \min\left(1, e^{-\frac{\Delta E}{k_B T}}\right)
$$

- **Temperatura alta ( $T$ elevata):** Esplora rapidamente l'entropia dello spazio, producendo prevalentemente conformazioni srotolate ed estese (*high-energy decoys*).

- **Temperatura bassa ( $T$ ridotta):** Tende a bloccarsi in minimi locali energetici a causa delle barriere entalpiche elevate, rendendo difficile raggiungere lo stato nativo a minima energia.

- **Uso per ML:** Utile per raccogliere campioni ad un ben definito regime termodinamico, ma richiede l'esecuzione di simulazioni separate a temperature differenti per coprire l'intero spettro conformazionale.

---

**DOMANDA**

Lavorando con il modello HP ovvero un sistema estremamente più semplice rispetto alla realtà , la formula della energia libera di gibbs può essere ancora presa in considerazione per l’analisi del sistema?

**Sì, assolutamente.** Il concetto di energia libera e le leggi della meccanica statistica rimangono perfettamente validi e matematicamente rigorosi anche nel modello HP.

Trattandosi di un sistema discreto su reticolo a volume costante, si utilizza formalmente l'**Energia Libera di Helmholtz** ( $F = E - TS$ ) anziché l'Energia Libera di Gibbs ( $G = H - TS$ ). Poiché nei modelli su reticolo non vi è espansione volumetrica o lavoro di pressione ( $P\Delta V = 0$ ), le due grandezze coincidono numericamente ( $F \equiv G$ ).

Nel modello HP, ciascun termine dell'equazione $F = E - TS$ assume un significato discreto ben definito:

1. **Energia Interna ( $E$ o $U$ ):**

   Rappresenta l'entalpia del sistema ed è calcolata contando i contatti non sequenziali tra monomeri idrofobici (H):

   $$
   E = -\epsilon \cdot N_{HH}
   $$

   dove $N_{HH}$ è il numero di coppie di residui H vicini di casa sul reticolo ma non adiacenti lungo la sequenza, ed $\epsilon > 0$ è l'energia di contatto (solitamente posta pari a $1$ come unità adimensionale).

2. **Entropia Conformazionale ( $S$ ):**

   Misura il grado di disordine strutturale della catena. È legata al numero di configurazioni distinte $\Omega(E)$ (passeggiate auto-evitanti o *Self-Avoiding Walks*) che la catena può assumere a un dato livello energetico $E$:

   $$
   S(E) = k_B \ln \Omega(E)
   $$

   Una proteina completamente srotolata possiede un'entropia $S$ immensa perché esistono tantissime conformazioni con $E = 0$. Lo stato nativo compatto possiede invece pochissime conformazioni accessibili ( $\Omega \to 1$ ), quindi un'entropia $S \approx 0$.

3. **Temperatura Ridotta ( $T^*$ ):**

   Nei modelli su reticolo la temperatura è espressa come grandezza adimensionale scalata rispetto all'energia di contatto:

   $$
   T^* = \frac{k_B T}{\epsilon}
   $$

---

### 2. Simulated Annealing (SA)

Inizia il campionamento a temperatura elevata $T_{\text{init}}$ per poi ridurla gradualmente verso $T_{\text{final}}$ secondo uno schema di raffreddamento (lineare, esponenziale o logaritmico).

- **Meccanismo:** Nelle fasi iniziali la catena esplora conformazioni ad alta entropia; raffreddandosi, le Pull Moves guidano la sequenza verso compattazioni idrofobiche sempre più stabili.

- **Uso per ML:** Estraendo conformazioni lungo l'intera traiettoria di raffreddamento si ottiene un *continuum* strutturale completo (da *random coil*, a stato *molten globule*, fino allo stato nativo o *near-native*).

### 3. Replica Exchange Monte Carlo (REMC / Parallel Tempering) (Approfondimento di seguito)

Simula contemporaneamente $M$ repliche del sistema a differenti temperature $T_1 < T_2 < \dots < T_M$. Ad intervalli regolari, si tenta uno scambio di conformazione tra repliche adiacenti $i$ e $j$ con probabilità:

$$
P_{\text{swap}} = \min\left(1, e^{(\beta_i - \beta_j)(E_i - E_j)}\right)
$$

dove $\beta = \frac{1}{k_B T}$.

- **Vantaggi:** Supera il problema del trappolamento nei minimi locali. Le conformazioni bloccate a bassa temperatura possono "risalire" ad alta temperatura, riorganizzarsi rapidamente tramite Pull Moves e poi tornare a raffreddarsi.

- **Uso per ML:** È la scelta d'elezione per la generazione di dataset. Raccoglie contemporaneamente stati nativi/a minima energia dalle repliche fredde e *decoys* di varia qualità dalle repliche intermedie e calde.

### 4. Campionamento Multicanonico / Wang-Landau (Flat-Histogram)

Anziché campionare secondo la distribuzione canonica di Boltzmann, stima iterativamente la densità degli stati $g(E)$ e modifica la probabilità di accettazione in modo che sia proporzionale a $\frac{1}{g(E)}$.

- **Meccanismo:** Forza la simulazione a compiere un *random walk* nello spazio delle energie, visitando ogni livello energetico con uguale frequenza.

- **Uso per ML:** Elimina il problema dello sbilanciamento delle classi (*class imbalance*). Nelle simulazioni standard gli stati ad energia intermedia sono immensamente più frequenti di quelli nativi; Wang-Landau garantisce un numero omogeneo di campioni per ogni fascia di energia $E$.

### 5. Campionamento Biasato basato su Variabili Collettive (Ispirato a Metadynamics)

Aggiunge al potenziale energetico un termine di penalità $V_{\text{bias}}(S)$ dipendente dalla storia della simulazione, dove $S$ è una variabile collettiva (CV) scalare o vettoriale.

- **Esempi di CV su reticolo HP:**

  - Numero di contatti idrofobici H-H non sequenziali $N_{\text{HH}}$ (proporzionale all'energia $E$ ).

  - Raggio di girazione $R_g$ (indice di compattezza).

  - Matrice delle distanze residue o sovrapposizione topologica rispetto a conformazioni già salvate.

- **Uso per ML:** Impedisce alla simulazione di ricampionare conformazioni già esplorate, spingendo la catena verso zone poco popolate dello spazio delle fasi per generare *decoys* strutturalmente diversificati.

### 6. Population Annealing Monte Carlo (PAMC)

Il **Population Annealing Monte Carlo (PAMC)** combina la strategia di raffreddamento progressivo del *Simulated Annealing* con la simulazione simultanea di un'intera **popolazione (o ensemble)** di $R$ repliche della proteina.

Mentre la temperatura del sistema viene gradualmente ridotta lungo uno schema di raffreddamento ( $\beta_0 < \beta_1 < \dots < \beta_N$, con $\beta = \frac{1}{k_B T}$ ), ad ogni incremento di $\beta$ l'algoritmo alterna due fasi fondamentali:

1. **Ricampionamento e Ribilanciamento (Resampling):** Ciascuna conformazione $i$ della popolazione viene duplicata o potata (eliminata) con un fattore proporzionale al rapporto dei pesi di Boltzmann tra la nuova e la precedente temperatura:

   $$
   \tau_i = \frac{e^{-(\beta_k - \beta_{k-1}) E_i}}{Q}
   $$

   dove $Q$ è una costante di normalizzazione per mantenere la dimensione complessiva della popolazione vicina al target $R$.

2. **Evoluzione MCMC locale:** Ogni copia sopravvissuta viene fatta evolvere autonomamente eseguendo un ciclo di mosse Monte Carlo locali (come le **Pull Moves**) alla temperatura corrente $T_k$. Questo passo ripristina l'equilibrio termodinamico locale e reintroduce diversità conformazionale.
- **Meccanismo:** A differenza del Simulated Annealing tradizionale (che segue una sola traiettoria), il resampling favorisce ed espande le conformazioni a bassa energia già identificate dai membri della popolazione, eliminando i rami poco promettenti ed evitando che il sistema rimanga bloccato in un singolo minimo locale.

- **Vantaggi:** È un algoritmo intrinsecamente e **massicciamente parallelizzabile** (ideale per l'esecuzione su GPU o cluster di calcolo), in quanto l'evoluzione MCMC delle $R$ repliche avviene in modo indipendente. Permette inoltre di stimare direttamente la funzione di partizione $Z(T)$ e l'energia libera del sistema lungo tutto il profilo di raffreddamento.

- **Uso per ML:** **Ottima.** In un'unica simulazione consente di salvare istantanee di centinaia o migliaia di conformazioni a vari livelli di temperatura. Ciò fornisce un dataset ampio e naturalmente stratificato: dalle conformazioni srotolate ad alta energia (*high-energy decoys*) registrate a $T$ elevate, fino ai *near-native decoys* e agli stati nativi stabili raccolti a basse $T$, completi di peso termodinamico.

## Tabella Comparativa Completa dei Modelli di Sampling

| **Metodo di Campionamento**     | **Copertura dello Spazio**            | **Efficienza per Stato Nativo** | **Rischio Minimi Locali** | **Idoneità per Dataset ML**                  |
| ------------------------------- | ------------------------------------- | ------------------------------- | ------------------------- | -------------------------------------------- |
| **Metropolis Canonico**         | Limitata alla $T$ scelta              | Molto Bassa (a $T$ bassa)       | Alto                      | Bassa (richiede più simulazioni)             |
| **Simulated Annealing**         | Traiettoria specifica                 | Buona                           | Medio                     | Media (focalizzato sulla convergenza)        |
| **Replica Exchange (REMC)**     | Ampia ed equilibrata                  | Eccellente                      | Molto Basso               | **Ottima** (standard di riferimento)         |
| **Wang-Landau**                 | Uniforme su tutte le $E$              | Buona                           | Assente                   | **Ottima** (evita *class imbalance*)         |
| **Metadynamics / Biased**       | Esplorazione guidata                  | Variabile                       | Assente                   | Buona (elevata varietà di *decoys*)          |
| **Population Annealing (PAMC)** | Molto ampia (sull'intera popolazione) | Eccellente                      | Molto Basso               | **Ottima** (massicciamente parallelizzabile) |

# Replica Exchange Monte Carlo (REMC)

L'algoritmo **Replica Exchange Monte Carlo (REMC)**, noto anche come *Parallel Tempering*, è un metodo ad ensemble esteso progettato per superare le barriere energetiche e prevenire l'intrappolamento in minimi locali durante la simulazione di sistemi complessi come il folding proteico.

Di seguito viene spiegato in dettaglio il funzionamento dell'algoritmo basandosi sull'articolo di **Thachuk et al. (2007)**.

### 1. Il Modello HP e lo Spazio delle Conformazioni

Nel modello HP (Hydrophobic-Polar) di Dill, la sequenza amminoacidica è semplificata in una stringa di due tipi di residui: **H** (idrofobici) e **P** (polari).

- La catena viene rappresentata come una passeggiata auto-evitante (*self-avoiding walk*) su un reticolo discreto (2D quadrato o 3D cubico).

- L'energia $E(c)$ di una conformazione $c$ è definita dal numero di contatti non sequenziali tra residui H (contatti topologici H-H), a ciascuno dei quali viene assegnato un valore di $-1$:

  $$
  E(c) = \sum_{j < k} N_{jk}
  $$

  dove $N_{jk} = -1$ se i residui $j$ e $k$ sono entrambi idrofobici (H) e adiacenti sul reticolo senza essere vicini lungo la sequenza, e $0$ altrimenti. **L'obiettivo è trovare conformazioni a minima energia (stato nativo)**.

### 2. Architettura dell'Algoritmo REMC

A differenza di una classica simulazione Monte Carlo a singola temperatura, il REMC mantiene e simula **simultaneamente $M$ repliche indipendenti** della stessa proteina.

1. **Inizializzazione delle Repliche**:

   - Vengono definite $M$ repliche, ciascuna associata a un valore di temperatura unico $T_1 < T_2 < \dots < T_M$.

   - Ogni replica parte da una conformazione valida della proteina (solitamente una configurazione iniziale causale o estesa).

2. **I Due Livelli dell'Algoritmo**:

   L'algoritmo alterna ciclicamente due fasi principali:

   - **Fase di Campionamento Locale (Local Monte Carlo Search)**

   - **Fase di Scambio di Temperatura (Replica Exchange)**

### 3. Fase 1: Campionamento Locale Monte Carlo

In questa fase, ciascuna replica $j$ esegue autonomamente e in parallelo una simulazione Monte Carlo alla propria temperatura $T_j$ per un numero prefissato di passi locali ($\phi$):

1. **Perturbazione della Struttura (Mossa)**:

   - Viene selezionato un residuo $k$ in modo casuale e si applica una mossa dal set di vicinato (es. *pull moves* o *VSHD moves*) per generare una nuova conformazione candidata $c'$.

2. **Criterio di Metropolis**:

   - Viene calcolata la differenza di energia $\Delta E = E(c') - E(c)$.

   - La mossa viene accettata secondo la probabilità di Metropolis:

     $$
     P(c \to c') = \begin{cases} 1 & \text{se } \Delta E \le 0 \\ e^{-\frac{\Delta E}{T_j}} & \text{se } \Delta E > 0 \end{cases}
     $$

     **Nota**: L'algoritmo metropolis classico prevede che, se una mossa aumenta la temperatura, essa venga accetata con probabilità  $e^{-\frac{\Delta E}{k_B T}}$. Come mai nel REMC si utilizza solo $e^{-\frac{\Delta E}{T_j}}$ senza il fattore $k_b$? La costante $k_B$ viene omessa perché si adottano unità ridotte adimensionali ponendo convenzionalmente $k_B = 1$, rendendo la temperatura $T$ un puro parametro algoritmico (adimensionale) di controllo del sistema. Infatti, nel modello HP l'energia $E$ non è espressa in unità del Sistema Internazionale (Joule) o chimiche (kcal/mol), ma come conteggio discreto dei contatti topologici H-H, dove ciascun contatto contribuisce con un valore adimensionale $\epsilon = -1$.

   - **Ruolo della Temperatura**:

     - **Temperature alte ($T_M$)**: L'accettazione di mosse che aumentano l'energia ($\Delta E > 0$) è frequente, permettendo alla catena di muoversi liberamente e superare barriere energetiche.

     - **Temperature basse ($T_1$)**: L'algoritmo accetta quasi esclusivamente mosse che riducono o mantengono invariata l'energia, concentrandosi sull'ottimizzazione e sulla ricerca del minimo locale/globale.

### 4. Fase 2: Scambio tra Repliche (Replica Exchange)

Ogni $\phi$ passi locali, si sospende il campionamento locale e si tenta lo scambio delle conformazioni (o delle temperature) tra repliche con temperature adiacenti $i$ e $j = i+1$.

1. **Calcolo della Probabilità di Scambio**:

   - Si valuta la probabilità di scambiare le configurazioni delle repliche $i$ e $j$ attraverso il criterio di Metropolis per l'ensemble esteso:

     $$
     P(\text{scambio}) = \begin{cases} 1 & \text{se } \Delta \le 0 \\ e^{-\Delta} & \text{se } \Delta > 0 \end{cases}
     $$

     dove:

     $$
     \Delta = (\beta_j - \beta_i) \cdot \left( E(c_i) - E(c_j) \right)
     $$

     con $\beta_k = \frac{1}{T_k}$ (inversa della temperatura).

2. **Accettazione dello Scambio**:

   - Se lo scambio è accettato, la conformazione $c_i$ passa alla temperatura $T_j$ e la conformazione $c_j$ passa alla temperatura $T_i$.

   - Gli scambi vengono eseguiti solitamente alternando coppie di indice pari e dispari per garantire la bilanciatura del processo Markoviano.

   **Riguardo allo scambio**: Quando in una simulazione Replica Exchange Monte Carlo (REMC) si hanno $M$ repliche disposte in ordine di temperatura ($T_1 < T_2 < \dots < T_M$), nasce un problema pratico: **una replica non può scambiare la propria posizione con due vicini contemporaneamente**. Ad esempio, la replica 2 non può essere scambiata nello stesso istante sia con la replica 1 che con la replica 3.

   Per evitare conflitti e garantire la correttezza statistica, si utilizza lo schema **Even-Odd (Pari-Dispari)**.

   Invece di scegliere coppie a caso, lo scambio tra repliche adiacenti viene diviso in due fasi alternate nel tempo, controllate da una variabile di scostamento (`offset`):

   - **Passo Dispari (Coppie Dispari/Pari — `offset = 0`)**:

     Si tentano gli scambi per le coppie aventi indice di partenza dispari:

     $$
     (1, 2), \quad (3, 4), \quad (5, 6), \quad \dots
     $$

     Nessuna replica appartiene a due coppie diverse contemporaneamente, quindi tutti questi scambi locali possono avvenire in parallelo o in sequenza senza interferenze.

   - **Passo Pari (Coppie Pari/Dispari — `offset = 1`)**:

     Al ciclo di scambio successivo, lo scostamento cambia e si tentano gli scambi per le coppie aventi indice di partenza pari:

     $$
     (2, 3), \quad (4, 5), \quad (6, 7), \quad \dots
     $$

   Alternando continuamente queste due fasi ad ogni iterazione di replica exchange (`offset = 1 - offset` ), ogni conformazione ha la possibilità di "viaggiare" lungo l'intera scala di temperature, salendo da $T_1$ a $T_M$ e scendendo di nuovo.

**Effetto dello Scambio**: Questo meccanismo permette a una conformazione rimasta intrappolata in un minimo locale a bassa temperatura di passare a una temperatura più alta, superare la barriera energetica, riorganizzarsi e successivamente ritornare a temperature basse verso conformazioni a energia ancora minore.

### 5. Set di Mosse (Move Sets) nell'Articolo

L'efficienza del REMC dipende in larga misura dal tipo di perturbazioni conformazionali utilizzate:

- **VSHD Moves** (*Verdier-Stockmayer-Hilhorst-Deutch*):
  Comprendono mosse tradizionali come *end moves* (rotazione degli estremi), *corner moves* (spostamento di angoli) e *crankshaft moves* (rotazione di strutture a 'U'). Hanno lo svantaggio di rimanere spesso bloccate in configurazioni molto compatte.

- **Pull Moves**:
  Introdotte da Lesh et al. (e generalizzate al reticolo 3D in questo lavoro):

  - Consentono di muovere un residuo in una posizione adiacente vuota e di "trascinare" in successione i residui vicini lungo la catena finché non si ripristina una conformazione valida.

  - Mantengono sempre la continuità e la validità della catena, garantendo una maggiore mobilità anche in stati ad alta densità/compattezza.

- **Modello Ibrido ($REMC_m$)**:
  Ad ogni passo Monte Carlo si sceglie una *pull move* con probabilità $p$ e una *VSHD move* con probabilità $1-p$.

### 6. Rilevanza per la Creazione del Dataset di Machine Learning

L'implementazione del REMC risponde perfettamente all'obiettivo del progetto:

1. **Generazione dello Stato Nativo**: Le repliche a temperatura più bassa ($T_1$) convergono efficacemente verso lo stato a minima energia o conformazione nativa.

2. **Generazione dei *Decoy* (Conformazioni Errate/Sub-ottimali)**: Le repliche a temperature medio-alte generano un insieme diversificato di conformazioni parzialmente ripiegate o completamente spiegate ad alta energia.

3. **Diversità Conformazionale**: L'articolo dimostra che il REMC con *pull moves* produce un'ampia ed eterogenea varieta di strutture, rendendo il dataset bilanciato e rappresentativo per l'addestramento del modello di ML.

## Come mai non è stato usato l'algoritmo metropolis-Hastings

 La probabilità di accettazione di una mossa nell'**algoritmo di Metropolis** si ricava imponendo la **simmetria della probabilità di proposta di una mossa** ovvero imponendo che la probabilità di proporre una mossa da una configurazione A alla configurazine B è la stessa che quella nel senso opposto da B ad A:

$$
q(A \to B) = q(B \to A)
$$

Nella realtà questa simmetria non si verifica sempre. Sarebbe quindi da usare la probabilità di accettaizone ricavata dall'algoritmo metropolis-hastings ovvero tenere conto della non simmetria: 

$$
q(A \to B) \neq q(B \to A)
$$

In particolare: 

- Se lo stato $A$ ha $n_A$ mosse ammissibili totali e lo stato $B$ ne ha $n_B$, la probabilità di scegliere la specifica mossa $A \to B$ è $q(A \to B) = \frac{1}{n_A}$.

- Per la mossa inversa, la probabilità è $q(B \to A) = \frac{1}{n_B}$.

- Se $n_A \neq n_B$, il rapporto delle probabilità di proposta $\frac{q(B \to A)}{q(A \to B)} = \frac{n_A}{n_B}$ non è pari a $1$.

Nel **Metropolis-Hastings completo**, la probabilità di accettazione considererebbe questo rapporto:

$$
\alpha(A \to B) = \min\left(1, \; \frac{n_A}{n_B} \cdot e^{-\frac{\Delta E}{T}}\right)
$$

Tuttavia, Thachuk et al. usano $n_A/n_B \approx 1$ ovvero,  viene adottata la regola classica di Metropolis. Questo accade per tre motivi fondamentali:

1. **Costo Computazionale del Calcolo di $n_B$**:

   Per calcolare il rapporto Hastings $\frac{n_A}{n_B}$, ad ogni mossa candidata $B$ bisognerebbe esplorare e contare preventivamente *tutte* le *pull moves* possibili a partire da $B$ prima ancora di sapere se la mossa verrà accettata. Questo raddoppierebbe o triplicherebbe il tempo di calcolo per ogni singolo passo Monte Carlo, rallentando drasticamente il campionamento.

2. **Obiettivo: Ottimizzazione e Ricerca del Minimo (Ground State)**:
   Nello studio del folding proteico sul modello HP (e nel progetto), il REMC viene utilizzato principalmente come **algoritmo di ottimizzazione euristica globale** e generatore di conformazioni. Il bilancio dettagliato rigoroso è indispensabile quando si devono calcolare medie d'ensemble termodinamiche ad altissima precisione, ma per trovare la conformazione nativa a minima energia e campionare decoy vicini, la lieve asimmetria nel numero di vicini non altera la capacità dell'algoritmo di convergere verso i minimi di energia.

3. **Selezione Uniforme del Residuo**: Consideriamo una singola proteina.  Mettiamoci nella fase dell'algoritmo in cui stiamo scegliendo su quale residuo della proteina vogliamo applicare una mossa. Invece di creare un elenco globale di *tutte* le mosse possibili per l'intera proteina e sceglierne una a caso, l'algoritmo inverte la logica:

   1. Sceglie prima un amminoacido $k$ tra $1$ e $n$ con probabilità perfettamente pari a $1/n$.

   2. Analizza solo le mosse ammissibili per *quel specifico* residuo $k$ e ne seleziona una.

   Se l'algoritmo scegliesse direttamente da un catalogo globale di tutte le mosse ammissibili della proteina, si verificherebbe un'asimmetria:

   - I residui situati in regioni molto flessibili (es. le estremità della catena, che hanno molte mosse valide) verrebbero estratti e mossi molto più spesso.

   - I residui situati al centro della catena o in zone ad alta densità (che hanno pochissime mosse valide) verrebbero quasi sempre ignorati.

   Selezionando prima il residuo $k$ in modo uniforme ($1/n$), si garantisce che **ogni amminoacido della sequenza riceva esattamente la stessa priorità temporale di tentativo**, indipendentemente da quanto sia libero o incastrato rispetto ai suoi vicini.

   
   Se l'algoritmo scegliesse la mossa estraendola direttamente dal catalogo di **tutte le mosse globalmente possibili** sulla proteina:

   - In una conformazione aperta $c$, ci sarebbero ad esempio $N(c) = 50$ mosse totali disponibili $\implies q(c \to c') = \frac{1}{50}$.

   - Nella nuova conformazione più compatta $c'$, ci sarebbero solo $N(c') = 10$ mosse totali disponibili $\implies q(c' \to c) = \frac{1}{10}$.

   In questo caso, il rapporto di Hastings sarebbe:

   $$
   \frac{q(c' \to c)}{q(c \to c')} = \frac{50}{10} = 5
   $$

   Un valore ben lontano da $1$. Ignorare questo fattore e usare Metropolis semplice farebbe fallire la correttezza teorica del campionamento.

   
   Nel codice dell'articolo, la probabilità di proporre una mossa viene scomposta in due parti:

   $$
   q(c \to c') = \underbrace{P(\text{scelta residuo } k)}_{1/n} \times \underbrace{P(\text{scelta mossa per } k)}_{1/m_k(c)}
   $$

   Quando calcoliamo il rapporto delle probabilità di proposta per andare da $c$ a $c'$ e viceversa:

   $$
   \frac{q(c' \to c)}{q(c \to c')} = \frac{\frac{1}{n} \cdot \frac{1}{m_{k'}(c')}}{\frac{1}{n} \cdot \frac{1}{m_k(c)}} = \frac{m_k(c)}{m_{k'}(c')}
   $$

   1. **La componente principale ($\frac{1}{n}$) si elide perfettamente**: Poiché ogni residuo ha la stessa identica probabilità ($\frac{1}{n}$) di essere scelto in qualsiasi conformazione, la fonte principale di asimmetria della catena viene eliminata alla radice.

   2. **La componente residua ($\frac{m_k(c)}{m_{k'}(c')}$) è trascurabile**: $m_k(c)$ è solo il numero di mosse locali per quel singolo residuo $k$ (che per le *pull moves* è quasi sempre un numero piccolissimo e costante, come $1$ o $2$, sia in $c$ che in $c'$).

   La **selezione uniforme del residuo ($1/n$) rende il rapporto di proposta $q(c' \to c) / q(c \to c') \approx 1$**.

### Implicazioni per il Progetto

Per la prima fase del progetto (ricerca dello stato nativo e generazione del dataset di *decoy*), la versione di **Metropolis standard** usata nell'articolo:

- Riduce la complessità di codice e i tempi di esecuzione.

- Mantiene un'elevata efficienza nel superare le barriere energetiche grazie al Replica Exchange.

- Genera un dataset vario e bilanciato, perfetto per il successivo addestramento del modello di Machine Learning.
