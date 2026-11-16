# Report — Decoy discriminator per il modello HP 2D

# 1. Introduzione

## 1.1 Contesto

Una proteina è una catena lineare di amminoacidi che, in ambiente fisiologico, si ripiega spontaneamente in una struttura tridimensionale specifica, detta **stato nativo**. Dalla struttura nativa dipende la funzione della proteina.

Secondo l'ipotesi termodinamica di Anfinsen, la struttura nativa è determinata dalla sola sequenza e corrisponde al minimo globale dell'energia libera. Prevedere il folding equivale quindi a cercare la conformazione di energia minima tra un numero di alternative che cresce in modo esponenziale con la lunghezza della catena.

Con modelli atomistici questa ricerca è molto costosa. Per studiare i principi generali del folding si usano allora **modelli semplificati su reticolo**, che riducono la proteina all'essenziale e rendono il problema trattabile al calcolatore.

## 1.2 Il modello HP

Il modello **HP** (idrofobico-polare), proposto da Dill nel 1985, è il più noto tra i modelli su reticolo. Si basa sull'osservazione che la forza principale del folding è l'effetto idrofobico. In questo contesto, i residui apolari tendono a raccogliersi all'interno della proteina, lontano dal solvente.

Nel modello ogni amminoacido è ridotto a uno di due tipi, **H** (idrofobico) o **P** (polare), e la proteina diventa una stringa come `PHHPPHPPHHPPHHPP`. Una conformazione è un cammino autoevitante sul reticolo quadrato 2D in cui residui consecutivi occupano nodi adiacenti e nessun nodo è occupato due volte.

L'energia della configurazione proteica viene calcolata contando i contatti H–H, cioè le coppie di residui H adiacenti sul reticolo ma non consecutivi nella catena:

```latex
E(c) = -\sum_{i < j - 1} \delta_{H}(i)\, \delta_{H}(j)\, \Delta(r_i, r_j)
```

dove δ_H(i) vale 1 se il residuo i è di tipo H, e Δ(r_i, r_j) vale 1 se le posizioni r_i e r_j sono adiacenti sul reticolo. Lo stato nativo è la conformazione con l'energia più bassa la quale, spesso, non è unica, perché più conformazioni degeneri possono raggiungere lo stesso minimo.

Nonostante la semplicità, trovare il minimo nel modello HP è un problema NP-difficile. Per questo si usano metodi stocastici come il **Replica Exchange Monte Carlo** (REMC), che in questo progetto segue la variante di Thachuk et al. (2007). ciao.

## 1.3 Obiettivi

L'obiettivo del progetto è capire se un algoritmo stocastico di ricerca del minimo che esplora il paesaggio energetico di una sequenza (REMC nel nostro caso), sia in grado di produrre un dataset valido per addestrare un modello di machine learning. Il modello, data in ingresso una sequenza HP e una sua conformazione, deve stabilire se la conformazione è **nativa** oppure un **decoy**.

Un decoy è una conformazione plausibile della sequenza che però non corrisponde allo stato nativo. Durante la ricerca il REMC visita sia le conformazioni di energia minima, che diventano gli esempi nativi, sia molte conformazioni a energia più alta, da cui si estraggono i decoy. Il dataset nasce quindi dall'esplorazione stessa del paesaggio energetico.

Il compito non è banale perché l'energia da sola non basta infatti lo stesso valore di E(c) può essere il minimo per una sequenza e lontano dal minimo per un'altra. Il modello deve quindi riconoscere lo stato nativo dalla sequenza e dalla forma della conformazione, senza eseguire una nuova ricerca del minimo.

In ingresso il modello riceve solo la sequenza e le coordinate della conformazione, non l'energia minima. Per classificare deve quindi ricavare due informazioni:

- **dalla sequenza**, quanto in basso può arrivare l'energia, in base al numero e alla posizione dei residui H lungo la catena;
- **dalla forma della conformazione**, se questa ha le caratteristiche di uno stato nativo, per esempio i residui H raccolti in un nucleo compatto.

Una volta addestrato, il modello risponde con un solo passaggio della rete, senza rieseguire il REMC su ogni nuova sequenza. È questo il vantaggio pratico rispetto al calcolo diretto dell'energia minima.

Il lavoro si articola in tre passi:

1. **Generazione del dataset.** Per 2000 sequenze HP casuali di 16 residui si esegue il REMC e si raccolgono le conformazioni native e un ugual numero di decoy, divisi in fasce di difficoltà in base alla distanza energetica dal minimo e affiancati da trappole cinetiche. L'energia minima trovata dal REMC è verificata per enumerazione esatta, così le etichette "nativo" sono certe.
2. **Modello di classificazione.** Il modello principale combina una rete neurale su grafo (GNN), applicata al grafo dei contatti della conformazione, con una rete convoluzionale 1D (CNN) applicata alla sequenza.
3. **Valutazione.** Il dataset è considerato valido se il modello generalizza a sequenze mai viste in addestramento. Le prestazioni sono confrontate con tre baseline di complessità crescente, da una regressione logistica sulla sola energia a un percettrone multistrato.

# 2. Metodi

## 2.1 Algoritmo REMC

### 2.1.1 Perché non basta un Monte Carlo semplice

Il REMC estende il **Monte Carlo Metropolis**, che esplora le conformazioni proteiche con piccole mosse locali. Una mossa che porta l'energia da E a E' viene accettata con probabilità:

```latex
P_{acc} = \min\left(1,\; e^{-(E' - E)/T}\right) \qquad (2)
```

dove T è la temperatura (con k_B = 1). Posto ΔE = E' − E, la probabilità vale quindi:

```latex
P_{acc} = \begin{cases} 1 & \text{se } \Delta E \le 0 \\ e^{-\Delta E / T} & \text{se } \Delta E > 0 \end{cases}
```

Il paesaggio energetico del modello HP è però **rugoso** avendo molti minimi locali separati da barriere. Il risultato dipende quindi molto dalla temperatura:

- **a T bassa** la catena si compatta in fretta, ma resta intrappolata nel primo minimo locale che incontra, per esempio una struttura compatta con il nucleo H nella posizione sbagliata;
- **a T alta** la catena supera le barriere, ma accetta troppe mosse sfavorevoli e non si assesta sul minimo.

Nessuna temperatura singola concilia le due esigenze. Il REMC risolve il problema facendo lavorare insieme più simulazioni a temperature diverse.

### 2.1.2 Repliche e scambi

Il REMC simula M copie della stessa sequenza, dette **repliche**, ognuna a una temperatura fissa. Le temperature sono equispaziate tra T_1 (la più fredda) e T_M (la più calda):

```latex
T_k = T_1 + (k - 1)\,\frac{T_M - T_1}{M - 1}, \qquad k = 1, \dots, M \qquad (3)
```

La simulazione procede per **cicli**. In ogni ciclo:

1. ogni replica esegue φ passi Metropolis alla propria temperatura, secondo l'equazione (2);
2. si propongono scambi di conformazione tra repliche a temperature adiacenti.

Ogni passo Metropolis propone una mossa locale su un residuo scelto a caso con probabilità ρ una **pull move** (Lesh et al., 2003), che resta possibile anche nelle conformazioni compatte, altrimenti una mossa **VSHD** (end, corner o crankshaft).

Lo scambio tra le repliche i e j, con energie E_i, E_j e temperature inverse β = 1/T, viene accettato con probabilità:

```latex
P_{swap} = \min\left(1,\; e^{(\beta_i - \beta_j)(E_i - E_j)}\right) \qquad (4)
```

In questo modo una conformazione trovata da una replica calda può scendere fino alla replica fredda, che la rifinisce verso il minimo. Al contrario, una conformazione intrappolata a bassa temperatura può salire, "sciogliersi" e uscire dal minimo locale.

### 2.1.3 Parametri

Per generare il dataset, il REMC viene eseguito una volta per ciascuna delle 2000 sequenze, partendo dalla catena distesa su un reticolo 20 × 20. Di seguito sono riportati i parametri di ingresso dell'algoritmo

|                        |         |                      |
| ---------------------- | ------- | -------------------- |
| Parametro              | Simbolo | Dataset del progetto |
| Numero di repliche     | M       | 5                    |
| Temperatura minima     | T_1     | 0,160                |
| Temperatura massima    | T_M     | 0,600                |
| Passi locali per ciclo | φ       | 100                  |
| Numero di cicli        | —       | 500                  |

Un seme diverso per ogni sequenza rende ogni esecuzione indipendente e riproducibile.

### 2.1.4 Verifica esatta

Il REMC è un metodo stocastico e non garantisce di trovare il minimo globale. Per le sequenze fino a 20 residui, quindi per tutte quelle del dataset, E_min viene confrontata con il minimo vero calcolato per **enumerazione esatta** di tutte le conformazioni. Una sequenza viene scartata in due casi:

- il REMC non ha raggiunto il minimo vero: i suoi "nativi" sarebbero in realtà decoy;
- E_min = 0, cioè nessuna conformazione forma contatti H–H e tutte sarebbero native.

Così l'energia di ogni esempio etichettato come nativo è certamente quella minima.

## 2.2 Generazione del dataset

Il dataset è formato da coppie (sequenza, conformazione) con un'etichetta binaria:

- **1** se la conformazione è nativa
- **0** se è un decoy.

Tutte le conformazioni provengono dalle esecuzioni del REMC descritte nella sezione 2.1.

Le sequenze sono 2000, di 16 residui, generate a caso in cui ogni residuo è H con probabilità 0,5. Una sequenza e la sua inversa descrivono la stessa catena letta dall'altro capo, quindi se ne tiene una sola delle due. Le sequenze scartate dalla verifica della sezione 2.1.4 vengono sostituite da nuove sequenze, finché non se ne ottengono 2000 valide.

Per ogni sequenza si raccolgono tutti gli stati visitati dal REMC:

- le **traiettorie**, cioè ogni conformazione accettata durante la simulazione;
- le **istantanee** di tutte le repliche (per ogni temperatura), salvate ogni 2 cicli.

Due conformazioni che differiscono solo per una traslazione, una rotazione o una riflessione del reticolo hanno la stessa energia e la stessa forma. Per questo ogni conformazione è ridotta a una **forma canonica** e gli stati equivalenti contano una volta sola.

### 2.2.1 Conformazioni native

Sono native tutte le conformazioni distinte che il REMC ha visitato all'energia minima E_min della sequenza. In questo modo il dataset contiene tutti gli stati **degeneri** trovati, non solo la migliore conformazione restituita dall'algoritmo.

Per non sbilanciare il dataset verso le sequenze molto degeneri, si tengono al massimo **20 nativi per sequenza**, scelti a caso quando il REMC ne ha trovati di più.

Grazie alla verifica della sezione 2.1.4, E_min coincide con il minimo vero calcolato per enumerazione esatta. Ne seguono due garanzie:

- ogni conformazione etichettata come nativa è davvero uno stato di energia minima;
- ogni decoy ha energia strettamente maggiore di E_min, quindi non può essere un nativo etichettato per errore.

Il REMC non garantisce invece di trovare **tutti** gli stati degeneri. Alcuni nativi possono quindi mancare dal dataset, ma nessun esempio ha un'etichetta sbagliata.

### 2.2.2 Decoy e fasce di difficoltà

I decoy sono conformazioni visitate dal REMC con energia maggiore di E_min. Per ognuno si calcola il **gap** dal minimo:

```latex
g(c) = E(c) - E_{min} \qquad (5)
```

Un decoy con gap piccolo è quasi nativo ed è difficile da distinguere mentre uno con gap grande è lontano dal minimo ed è facile. Lo stesso gap però pesa in modo diverso a seconda della sequenza: un gap di 2 è grande se E_min = −3, piccolo se E_min = −9. Per questo le fasce si basano sul gap **relativo** g / |E_min|, che vale 0 per un nativo e 1 per una conformazione senza contatti H–H.

|        |                            |            |
| ------ | -------------------------- | ---------- |
| Fascia | Gap relativo g / \|E_min\| | Difficoltà |
| near   | fino a 0,34 (oppure g = 1) | alta       |
| mid    | da 0,34 a 0,67             | media      |
| far    | oltre 0,67                 | bassa      |

Ogni sequenza riceve tanti decoy quanti sono i suoi nativi, divisi in parti uguali tra le tre fasce. Se una fascia non ha abbastanza candidati, i decoy mancanti vengono presi dalle altre. Dentro ogni fascia i decoy sono scelti a caso.

### 2.2.3 Trappole cinetiche

Una **trappola cinetica** è un minimo locale dell'energia rispetto all'insieme delle mosse: nessuna singola mossa, VSHD o pull move, abbassa l'energia della conformazione. Indicato con N(c) l'insieme delle conformazioni raggiungibili da c con una mossa, c è una trappola se:

```latex
E(c') \ge E(c) \quad \forall\, c' \in N(c) \qquad (6)
```

Sono gli stati in cui un Monte Carlo a bassa temperatura resta bloccato. Di solito sono conformazioni compatte, con molti contatti H–H ma con il nucleo idrofobico nella posizione sbagliata. Per il modello sono i decoy più difficili, perché somigliano a uno stato nativo.

Per questo metà della quota della fascia near è riservata alle trappole. Le trappole si cercano tra i primi 200 candidati near di ogni sequenza. Se non ce ne sono abbastanza, la quota viene completata con altri decoy near.

Ogni decoy del dataset porta un indicatore che dice se è una trappola, così nei risultati si può misurare a parte quanto il modello sbaglia su questi casi.

### 2.2.4 Bilanciamento e suddivisione

Il bilanciamento è **1:1 per sequenza**, non solo sul totale. Se una sequenza avesse più nativi che decoy, il modello potrebbe imparare che quella sequenza è "spesso nativa" invece di guardare la conformazione.

La suddivisione in train, validation e test (80%, 10%, 10%) è fatta **per sequenza** ovvero tutte le conformazioni di una sequenza finiscono nello stesso insieme. Il test contiene quindi 200 sequenze mai viste in addestramento, e misura proprio la capacità di generalizzare richiesta dagli obiettivi. Una suddivisione per conformazione metterebbe invece nativi della stessa sequenza sia in train sia in test, e il modello potrebbe riconoscerli a memoria.

## 2.3 Modello di classificazione

Il classificatore riceve una sequenza HP e una sua conformazione e restituisce la probabilità che la conformazione sia nativa. Una conformazione c è nativa se la sua energia coincide con l'energia minima della sequenza, cioè se E(c) = E_min.

Data una sequenza di ingresso, il modello deve stimare:

- **L'energia della configurazione (E(c)).** Questa è una proprietà *locale* della conformazione e dipende solo dai contatti H–H presenti;
- **L'energia minima della sequenza (E_min)**. Essa una proprietà *globale* della sequenza e dipende solo dalla stringa HP.

Il modello riflette questa scomposizione con due rami separati, uno per la conformazione e uno per la sequenza, i cui risultati vengono combinati da un classificatore finale.

### 2.3.1 Rappresentazione dell'input

Ogni esempio fornisce al modello due sole informazioni: la sequenza HP e la conformazione.

**Sequenza.** La sequenza di N = 16 residui è codificata come vettore binario h, con h_i = 1 se il residuo i è di tipo H. A ogni residuo sono associate quattro caratteristiche calcolate dalla sola sequenza:

- il tipo (H o P);
- la **parità** dell'indice i;
- la posizione lungo la catena, normalizzata come i / (N − 1);
- un indicatore che vale 1 per i due estremi della catena.

**Conformazione.** Le coordinate dei residui sul reticolo non sono usate direttamente ma vengono convertite nella **mappa dei contatti** C, una matrice N × N con C_ij = 1 se i residui i e j sono adiacenti sul reticolo ma non consecutivi nella catena. Questa rappresentazione è invariante per traslazioni, rotazioni e riflessioni della conformazione, che non ne cambiano l'energia.

La conformazione diventa così un **grafo** in cui i nodi sono i residui e gli archi sono di due tipi:

- i **legami** tra residui consecutivi (i, i + 1), uguali per tutte le conformazioni;
- i **contatti** topologici dati da C, che distinguono una conformazione dall'altra.

### 2.3.2 Architettura GNN + CNN 1D

Il modello principale ha due rami, uno per ciascun sotto-problema, e tre teste di uscita. Entrambi i rami partono dalle quattro caratteristiche per residuo descritte in 2.3.1.

[embed: node/0665bacf-6ccd]

**Ramo della conformazione (GNN).** Una rete neurale su grafo (Graph Neural Network) elabora il grafo della conformazione. Ogni residuo i è descritto da un vettore h_i di dimensione 64, inizializzato con una trasformazione lineare delle sue caratteristiche. In ciascuno dei 4 strati di *message passing* ogni residuo riceve un messaggio da ogni vicino, con pesi diversi per legami e contatti:

```latex
m_i = \sum_{j} B_{ij} \, W_b \, h_j + \sum_{j} C_{ij} \, W_c \, h_j
```

dove B è la matrice dei legami e C la mappa dei contatti. Il vettore del residuo viene poi aggiornato combinando il suo valore con la somma dei messaggi:

```latex
h_i \leftarrow \mathrm{LayerNorm}\left(h_i + \mathrm{MLP}\left([\,h_i,\; m_i\,]\right)\right)
```

Dopo 4 strati ogni residuo ha raccolto informazioni dai residui a distanza fino a 4 nel grafo. I vettori dei residui vengono infine riassunti in un unico vettore **z_conf** concatenando la loro somma e il loro massimo.

**Ramo della sequenza (CNN 1D).** Una rete convoluzionale 1D elabora la sola sequenza, senza vedere la conformazione. È composta da 4 strati convolutivi con filtri di ampiezza 5 e attivazione ReLU. Ogni residuo vede così 4 · (5 − 1) + 1 = 17 residui, cioè l'intera catena di 16. Lo stesso riassunto somma + massimo produce il vettore **z_seq**.

**Classificatore e teste ausiliarie.** Un percettrone multistrato (MLP) riceve z_conf, z_seq e il loro prodotto elemento per elemento, che permette di confrontare direttamente le due rappresentazioni. Produce due uscite:

- il **logit** della classe, da cui la probabilità di conformazione nativa P = σ(logit);
- una stima del **gap** E(c) − E_min.

Due teste ausiliarie, attive solo durante l'addestramento, stimano l'energia E(c) dal solo z_conf ed E_min dal solo z_seq. Servono a guidare ciascun ramo verso il proprio sotto-problema. Il ramo della conformazione impara a riconoscere l'energia, quello della sequenza a stimarne il minimo.

|                                 |        |
| ------------------------------- | ------ |
| Iperparametro                   | Valore |
| Dimensione dei vettori nascosti | 64     |
| Strati della GNN                | 4      |
| Strati della CNN                | 4      |
| Ampiezza dei filtri della CNN   | 5      |
| Dropout nelle teste MLP         | 0,1    |

### 2.3.3 Baseline

Le baseline sono modelli più semplici di quello principale, usati come termine di confronto. Ognuna riceve un'informazione in più della precedente e confrontandone le prestazioni si capisce quale informazione fisica serve davvero per riconoscere lo stato nativo e quanto aggiunge il modello principale.

**B0a — solo energia.** Verifica se l'energia della conformazione basta, da sola, a distinguere una conformazione nativa da un decoy. Calcola E(c) dai contatti H–H e la usa come unico criterio. Più l'energia è bassa, più la conformazione è considerata nativa. Ci si aspetta che sbagli spesso, perché lo stesso valore di E(c) può essere il minimo per una sequenza e non per un'altra (vedi 1.3).

**B0b — energia e composizione della sequenza.** Verifica se semplici proprietà della sequenza bastano a stimare quanto in basso può arrivare l'energia. B0a non sa quanto in basso può arrivare l'energia di una sequenza mentre B0b prova a stimarlo dalla sua composizione. Sul reticolo quadrato un contatto unisce sempre un residuo di indice pari a uno di indice dispari, quindi il numero massimo di contatti H–H, e con esso E_min, dipende da quanti H si trovano in posizione pari e in posizione dispari. B0b usa questi conteggi, insieme a E(c), come stima approssimativa di E_min. I conteggi danno infatti solo un limite massimo ai contatti mentre quanti se ne formano davvero dipende dall'ordine degli H lungo la catena, quindi sequenze con gli stessi conteggi possono avere E_min diverse.

**B1b — contatti senza struttura a grafo.** Verifica se conta il modo in cui l'informazione viene elaborata. Riceve la sequenza e i contatti H–H, cioè le stesse informazioni del modello principale, ma le elabora con un percettrone multistrato (MLP), cioè una rete neurale semplice che ignora la struttura della catena. Se il modello principale fa meglio di B1b, il vantaggio viene dall'architettura e non da un'informazione in più.

## 2.4 Addestramento e metriche

14 nov 2026 · @Alessandro

Il modello è addestrato minimizzando la *binary cross-entropy* tra la probabilità prevista e l'etichetta vera (nativa o decoy). Alla perdita si sommano, con peso 0,1, gli errori quadratici delle teste ausiliarie su E(c), E_min e gap. Queste grandezze servono solo come bersagli poichè in ingresso il modello riceve sempre e soltanto la sequenza e la mappa dei contatti.

Una catena letta dall'estremo opposto è lo stesso stato fisico, con la stessa energia. Per sfruttare questa simmetria, in ogni batch metà degli esempi viene invertita a caso.

I pesi sono ottimizzati con Adam. Dopo ogni epoca si misura l'AUROC sul validation set. L'addestramento si ferma dopo 10 epoche senza miglioramenti e si tengono i pesi migliori. Il test set è usato una sola volta, alla fine. Le baseline sono valutate allo stesso modo.

Le prestazioni sono misurate con due metriche:

- **AUROC**: la probabilità che una conformazione nativa scelta a caso riceva un punteggio più alto di un decoy scelto a caso. Vale 1 per una separazione perfetta e 0,5 per un classificatore casuale.
- **Accuratezza**: Rappresenta la **frazione di esempi classificati correttamente**. Si è impostata una soglia a 0,5 complessiva e per fascia (near, mid, far e trappole cinetiche). Mostra se gli errori si concentrano sui decoy con energia vicina al minimo.

# 3. Risultati

Il dataset prodotto dal REMC è valido. Il modello GNN + CNN 1D classifica correttamente il **98,1%** delle conformazioni di test (AUROC 0,999), che appartengono a 200 sequenze mai viste in addestramento. La regressione logistica sulla sola energia si ferma al 79,3%.

I risultati seguono l'ordine del lavoro. Prima si verifica la qualità del dataset (3.1), poi si confronta il modello con le baseline (3.2), infine si guarda dove ciascun modello sbaglia (3.3). Tutte le metriche si riferiscono al test set, salvo dove indicato.

## 3.1 Esito della verifica esatta e statistiche del dataset

Il REMC ha raggiunto il minimo esatto in **tutte** le sequenze: nessuna è stata scartata per un minimo mancato. Su 2013 sequenze generate, le sole 13 scartate avevano E_min = 0, cioè nessun contatto H–H possibile.

Le 2000 sequenze rimaste sono divise per sequenza tra train, validation e test, con nativi e decoy in rapporto 1:1.

|            |          |            |            |               |
| ---------- | -------- | ---------- | ---------- | ------------- |
| Split      | Sequenze | Nativi     | Decoy      | Esempi totali |
| Train      | 1600     | 25 177     | 25 177     | 50 354        |
| Validation | 200      | 3 028      | 3 028      | 6 056         |
| Test       | 200      | 3 115      | 3 115      | 6 230         |
| **Totale** | **2000** | **31 320** | **31 320** | **62 640**    |

Per ogni sequenza il REMC visita in mediana 25 917 stati distinti (da 8 005 a 103 836). Molte sequenze sono fortemente degeneri: gli stati nativi trovati sono 37 in mediana e arrivano a 46 190. Per questo i nativi sono limitati a 20 per sequenza, tetto raggiunto da 1293 sequenze (64,7%); in media una sequenza fornisce 15,7 nativi.

I decoy sono distribuiti in modo quasi uniforme tra le fasce: 12 095 near (38,6%), 9 953 mid (31,8%) e 9 272 far (29,6%). Il 34,5% dei decoy (10 810) è anche una trappola cinetica. Più della metà dei decoy (55,5%) dista dal minimo solo 1 o 2 unità di energia, quindi il dataset è sbilanciato verso i casi difficili.

<img src="/home/ale/.config/marktext/images/d6093b48a13c889dec36a74d928265e1cb73716a.png" title="" alt="" width="523" />

Come si può vedere, le energie estreme sono rare infatti solo 4 sequenze hanno E_min = −9 e 9 hanno E_min = −1. Le metriche calcolate per queste energie si basano quindi su pochi esempi.

## 3.2 Confronto tra modello e baseline

Il modello GNN + CNN 1D supera tutte e tre le baseline: 98,1% di accuratezza sul test, contro il 96,5% della migliore baseline (B1b) e il 79,3% della sola energia (B0a).

|                  |                                                  |             |           |                    |
| ---------------- | ------------------------------------------------ | ----------- | --------- | ------------------ |
| Modello          | Input usato                                      | Accuratezza | AUROC     | AUROC per sequenza |
| **GNN + CNN 1D** | grafo dei contatti + sequenza                    | **98,1%**   | **0,999** | 1,000              |
| B1b (MLP)        | contatti H–H + sequenza, senza struttura a grafo | 96,5%       | 0,995     | 1,000              |
| B0b (logistica)  | energia + 5 caratteristiche della sequenza       | 92,0%       | 0,975     | 1,000              |
| B0a (logistica)  | sola energia                                     | 79,3%       | 0,871     | 1,000              |

**AUROC per sequenza.** Vale 1 per tutti i modelli, B0a compresa. Questo indica che dentro una stessa sequenza i nativi hanno sempre energia più bassa dei decoy, e l'energia basta a ordinarli. Questa metrica non distingue quindi i modelli. La difficoltà sta nel confronto tra sequenze diverse.

**AUROC.** Su tutte le sequenze insieme sale da 0,871 (B0a) a 0,999 (GNN), quindi i modelli migliori stimano meglio E_min dalla sequenza. La GNN lo stima con un errore medio assoluto di 0,27 unità. Dopo ogni epoca il modello viene valutato sul validation set calcolando l'AUROC. Se l'AUROC non migliora per 10 epoche di fila (patience = 10 in config/default.toml), l'addestramento si ferma e si tengono i pesi dell'epoca migliore. Il massimo arriva all'epoca 24, quindi l'addestramento si ferma alla 34 (24 + 10). L'AUROC è quindi anche il criterio che sceglie il modello finale.

**Accuratezza.** Le poche caratteristiche della sequenza aggiunte da B0b valgono 12,7 punti rispetto a B0a, e la GNN arriva al 98,1%. L'accuratezza della GNN passa dal 99,4% in validation al 98,1% sul test, che contiene solo sequenze mai viste. Il calo è di appena 1,3 punti, quindi il modello generalizza bene a sequenze nuove.

## 3.3 Prestazioni per fascia di difficoltà

Le fasce near e le trappole cinetiche separano i modelli. La GNN le classifica correttamente nel 98% dei casi, mentre B0a scende al 44% sui near, peggio del caso. I decoy mid e far sono invece facili per tutti infatti anche la sola energia ne riconosce il 91–100%.

<img src="/home/ale/.config/marktext/images/1137a2b8383e808f8d3f72ed901b103556dd082f.png" title="" alt="" width="553" />

Il calo di B0a sui near è atteso. Un decoy near ha energia vicina al minimo della propria sequenza, e senza stimare E_min un modello non può distinguerlo da un nativo di un'altra sequenza. La GNN mantiene un'accuratezza tra il 96,3% e il 100% per ogni valore di E_min, quindi non dipende da quanto è profondo il minimo.

Sui 6230 esempi di test la GNN sbaglia 117 volte con 95 nativi classificati come decoy e 22 decoy near classificati come nativi. Gli errori sono concentrati in 23 delle 200 sequenze di test, e le 5 sequenze peggiori ne raccolgono 74 (63%). Sui nativi rifiutati la GNN stima in media un E_min più basso di 0,39 unità rispetto a quello vero. Il modello crede cioè che la sequenza possa scendere più in basso, e per questo giudica il vero nativo un decoy. Un esempio è `PPPPHPPHHHHHHHHH`, con E_min = −5 stimato a −5,42.

# 4. Discussione e limiti

I risultati rispondono in modo affermativo alla domanda se il REMC sia in grado di produrre un dataset con cui addestrare un classificatore capace di generalizzare a sequenze nuove. Tuttavia, i limiti sono da tenere in considerazione

## 4.1 Il REMC produce un dataset valido?

Sì, per catene di 16 residui. La validità poggia su tre fatti, ognuno legato a una parte dei risultati.

**Le etichette sono corrette.** Il REMC ha raggiunto il minimo esatto in tutte le 2013 sequenze generate (sezione 3.1). Con 5 repliche e 500 cicli la ricerca è quindi più che sufficiente per N = 16. Grazie alla verifica esatta, nessun decoy può essere un nativo etichettato per errore.

**I decoy sono difficili.** Il 55,5% dei decoy dista dal minimo solo 1 o 2 unità di energia e il 34,5% è una trappola cinetica. Questi esempi nascono proprio dall'esplorazione del paesaggio energetico infatti sono gli stati in cui il REMC stesso passa prima di arrivare al minimo. Un dataset di conformazioni casuali conterrebbe soprattutto decoy lontani, facili da scartare.

**Il modello generalizza.** Tutte le conformazioni di una sequenza stanno nello stesso insieme, quindi le 200 sequenze di test sono del tutto nuove. Il modello non può quindi ricordarne i nativi e deve stimare E_min da solo. Su queste sequenze l'accuratezza è del 98,1%, appena 1,3 punti sotto il validation (Se il modello fosse adattato al validation per caso, sul test crollerebbe) RIMUOVI.

Va però precisato che la certezza delle etichette viene dall'enumerazione esatta, non dal REMC da solo.

## 4.2 Che cosa impara il modello

Il problema si riduce quasi del tutto a stimare E_min dalla sequenza. Lo mostra l'AUROC per sequenza, che vale 1 per tutti i modelli, compresa B0a. Dentro una stessa sequenza l'energia basta a ordinare nativi e decoy. La difficoltà nasce solo nel confronto tra sequenze diverse infatti se, per una certa sequenza, E(c) = −4, questo rappresenta un nativo se E_min = −4, ma un decoy se E_min = −7. Senza stimare E_min il modello non può decidere.

La scala delle baseline indica quale informazione serve davvero:

- **B0a → B0b (+12,7 punti).** Pochi conteggi degli H in posizione pari e dispari bastano a stimare in modo grezzo quanto in basso può arrivare l'energia. È il guadagno più grande della scala.
- **B0b → B1b (+4,5 punti).** Conoscere la sequenza completa e i contatti H–H, non solo i conteggi, rende la stima di E_min più precisa.
- **B1b → GNN + CNN 1D (+1,6 punti).** A parità di informazione, l'architettura che segue la catena aggiunge un guadagno piccolo ma netto. Si vede soprattutto sui decoy near e sulle trappole, le fasce che separano i modelli (sezione 3.3).

Il modello principale si comporta quindi come uno stimatore di E_min appreso, affiancato dal confronto con l'energia della conformazione. L'errore medio di 0,27 unità su E_min è coerente con questa lettura.

## 4.3 Dove il modello sbaglia

Gli errori della GNN sono pochi, sbilanciati e concentrati. Dei 117 errori sul test, 95 sono nativi rifiutati e solo 22 sono decoy accettati. Il modello tende quindi a essere prudente assegnando l'etichetta di decoy in caso di incertezza.

La causa è la stessa individuata nella 4.2. Sui nativi rifiutati il modello stima un E_min più basso di quello vero, in media di 0,39 unità. Crede cioè che la sequenza possa compattarsi di più, e giudica il vero nativo un decoy.

Gli errori non sono distribuiti a caso. Cadono in 23 delle 200 sequenze di test e le 5 peggiori ne raccolgono il 63%. Il caso di `PPPPHPPHHHHHHHHH`, con nove H consecutivi, suggerisce che si tratti di sequenze poco rappresentate nel train. Con H scelto con probabilità 0,5, blocchi così lunghi sono rari, e il ramo della sequenza ha pochi esempi da cui imparare il loro E_min.

Le trappole cinetiche, che dovevano essere i decoy più difficili, non mettono in crisi la GNN, che le classifica correttamente nel 98% dei casi. Restano invece decisive per separare i modelli. Per esempio, B0a, che non stima E_min, scende al 44% sui near.

Infine, lo sbilanciamento tra i due tipi di errore (nativi classificati come decoy, 95; e decoy classificati come nativi, 22) dipende anche dalla soglia fissa di classificazione tra nativi e decoy a 0,5. Con un AUROC di 0,999 i punteggi separano quasi perfettamente le classi, quindi una soglia scelta sul validation set potrebbe ridurre i nativi rifiutati.

## 4.4 Limiti

I risultati valgono per le condizioni in cui sono stati ottenuti, e alcune di queste ne limitano la portata.

- **Modello semplificato.** Il reticolo quadrato 2D e l'alfabeto di due lettere sono molto lontani da una proteina reale. Il progetto studia i principi del folding, non prevede strutture vere.
- **Lunghezza fissa.** Tutte le sequenze hanno 16 residui. Non si sa come il modello si comporti su catene di altra lunghezza, che avrebbero E_min diverse e grafi più grandi.
- **Verifica esatta limitata.** L'enumerazione esatta è possibile solo fino a circa 20 residui. Per catene più lunghe le etichette dipenderebbero dal solo REMC, che non garantisce il minimo globale, e alcuni "nativi" potrebbero essere decoy.
- **Stati degeneri incompleti.** Il REMC non trova necessariamente tutte le conformazioni di energia minima, e il tetto di 20 nativi è stato raggiunto dal 64,7% delle sequenze. Il dataset copre quindi solo una parte degli stati nativi possibili.
- **Sequenze casuali.** Le sequenze sono generate a caso e sono spesso molto degeneri, con 37 stati nativi in mediana. Le proteine naturali, selezionate dall'evoluzione, hanno di solito un'unica struttura nativa ben definita.
- **Decoy dello stesso generatore.** Nativi e decoy provengono dallo stesso REMC. Il modello potrebbe comportarsi peggio su decoy prodotti in altro modo, per esempio da un altro algoritmo di ricerca.
- **Vantaggio pratico ridotto a N = 16.** Per catene così corte l'enumerazione esatta è già abbastanza rapida, quindi un modello che evita il REMC serve più come prova di principio che come strumento. Il vantaggio diventerebbe concreto solo su catene più lunghe, dove però valgono i limiti precedenti.

# 5. Conclusioni e sviluppi futuri



## 5.1 Conclusioni

Il progetto si chiedeva se un algoritmo stocastico di ricerca del minimo, il REMC, fosse in grado di produrre un dataset valido per addestrare un classificatore di conformazioni native (sezione 1.3). Per catene HP di 16 residui sul reticolo quadrato 2D la risposta è **sì**.

Il REMC ha raggiunto il minimo esatto in tutte le 2013 sequenze generate. Dalla sua esplorazione del paesaggio energetico sono nati 62 640 esempi bilanciati, con decoy in gran parte vicini al minimo infatti il 55,5% dista solo 1 o 2 unità di energia e il 34,5% è una trappola cinetica. Sono proprio questi i casi che rendono il compito non banale.

Il modello GNN + CNN 1D addestrato su questo dataset classifica correttamente il **98,1%** delle conformazioni di 200 sequenze mai viste (AUROC 0,999). Supera tutte le baseline, dal 79,3% della sola energia al 96,5% dell'MLP, e mantiene il 98% anche sui decoy near e sulle trappole.

Il confronto con le baseline mostra che il compito si riduce quasi del tutto a **stimare E_min dalla sequenza**. Dentro una stessa sequenza l'energia basta a ordinare nativi e decoy. La difficoltà sta nel confronto tra sequenze diverse. Il guadagno più grande viene da poche informazioni sulla composizione della sequenza (+12,7 punti), mentre l'architettura a grafo aggiunge un margine piccolo ma netto (+1,6 punti).

Una volta addestrato, il modello giudica una conformazione con un solo passaggio della rete, senza rieseguire il REMC. A N = 16 questo vantaggio resta una prova di principio, perché l'enumerazione esatta è ancora rapida, ma indica una strada per catene più lunghe.

## 5.2 Sviluppi futuri

Gli sviluppi più utili rispondono ai limiti della sezione 4.4 e agli errori della sezione 4.3.

- **Catene più lunghe e di lunghezza variabile.** La GNN lavora su grafi di qualsiasi dimensione, quindi può essere addestrata su sequenze di lunghezze diverse. Oltre i 20 residui, dove l'enumerazione esatta non è più praticabile, le etichette andrebbero controllate in altro modo, per esempio con più esecuzioni indipendenti del REMC o con sequenze di riferimento di cui è noto il minimo. È qui che il modello darebbe un vantaggio pratico rispetto alla ricerca diretta.
- **Reticoli diversi.** Lo stesso schema si può applicare al reticolo cubico 3D . Basta adattare le mosse del REMC e la costruzione della mappa dei contatti.
- **Sequenze poco rappresentate.** Gli errori si concentrano in poche sequenze con lunghi blocchi di H, rare quando ogni residuo è H con probabilità 0,5. Si potrebbero generare più sequenze di questo tipo o variare la probabilità di H, così che il ramo della sequenza veda più esempi da cui stimare E_min.
- **Decoy da altri generatori.** Per verificare che il modello non dipenda dal REMC, lo si può valutare su decoy prodotti in altro modo, per esempio da un Monte Carlo a temperatura singola o da conformazioni compatte casuali.
- **Il modello come supporto alla ricerca.** La stima di E_min data dal ramo della sequenza può servire al REMC stesso, per esempio come criterio per fermare la simulazione quando una conformazione raggiunge l'energia prevista. Il classificatore passerebbe così da giudice a parte dell'algoritmo di ricerca.
