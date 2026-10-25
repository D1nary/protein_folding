# Architettura ML

## Cosa deve imparare il modello

Il modello deve analizzare una configurazione di ingresso e dire se quella configurazione è un nativo o no. In altre parole, verifica l'uguaglianza tramite l'etichetta `label = [E(seq, conf) == E_min(seq)]` restituendo 1 se l'uguaglianza viene rispettata.

Il compito si divide quindi in due parti molto diverse:

1. **Calcolo dell'energia della configurazione E(seq, conf)**: è una funzione esatta e locale. È meno il numero di contatti H-H, cioè la somma sulle coppie di lattice vicine (non consecutive) di H_i·H_j. Si calcola facilmente se il modello vede i contatti.
2. **Calcolo del minimo reale della sequenza E_min(seq)**: dipende solo dalla sequenza, ma è globale e difficile da stimare.  Il modello impara a **stimare un numero**, l’energia minima per quella sequenza. Non impara la **forma** a energia minima.

Una volta eseguiti questi due calcoli, la rete può effettuare l'uguaglianza vista prima.

Gli esempi facili sono i decoy `far`, perché sono forme aperte. Quelli difficili sono i `near`, con gap 1: per riconoscerli il modello deve sapere che per *quella* sequenza si poteva fare un contatto in più.

## Schema e funzionamento generale del modello

Le due parti del compito che il modello deve eseguire sono gestite da due reti diverse ma che lavorano in parallelo. In particolare, la gestione del calolo dell'energia della configurazione è affidata ad una **Graph Neural Network (GNN)** mentre la parte del calcolo dell'energia minima della configurazione è affidata a una  **CNN 1D**

Le fasi dell'addestramento del modello sono:

**1. Ingresso**

- Vengono passate in ingresso la sequenza e la configurazione.

**2. Ramo conformazione**

- La sequenza e la configurazione vengono trasformate in un grafo: nodi = residui (H/P), archi = legami e contatti.
- La GNN genera un vettore `z_conf` con le informazioni sulla configurazione.
- Da `z_conf` lo **stimatore di energia** predice l'energia della configurazione `Ê`.

**3. Ramo sequenza** (in parallelo al punto 2)

- Riceve solo la sequenza, senza la configurazione.
- Genera un vettore `z_seq` con le informazioni sulla sequenza.
- Da `z_seq` lo **stimatore di energia minima (MLP)** predice l'energia minima della sequenza `Ê_min`.

**4. Classificatore**

- Riceve `z_conf` e `z_seq` e li confronta.
- Predice la probabilità che la configurazione sia nativa `p̂` e il gap `ĝ`.

**5. Calcolo della loss**

- Ogni predizione viene confrontata con il proprio target:

  - `Ê` con `energy` → `MSE(Ê, energy)`;
  - `Ê_min` con `e_min` → `MSE(Ê_min, e_min)`;
  - `p̂` con `label` → `BCE(p̂, label)`;
  - `ĝ` con `gap` → `MSE(ĝ, gap)`.

- Questi termini si sommano in un'unica loss totale.

**6. Aggiornamento dei pesi**

- Stimatore di energia e GNN: aggiornati da `MSE(Ê, energy)`.
- Stimatore di energia minima e ramo sequenza: aggiornati da `MSE(Ê_min, e_min)`.
- Classificatore, GNN e ramo sequenza: aggiornati da `BCE(p̂, label)` e `MSE(ĝ, gap)`.

Questi 6 passi si ripetono per ogni batch di esempi, per molte epoche. Dopo il training si usa solo `p̂`, l'uscita del classificatore.

| Testa                             | Riceve             | Predice                            | Target         |
| --------------------------------- | ------------------ | ---------------------------------- | -------------- |
| **stimatore di energia**          | `z_conf`           | `Ê`                                | `energy`       |
| **stimatore di energia minima**   | `z_seq`            | `Ê_min`                            | `e_min`        |
| **classificatore** (testa finale) | `z_conf` e `z_seq` | probabilità di nativo `p̂`, gap `ĝ` | `label`, `gap` |

**Multi-task.** I target continui del dataset (`energy`, `e_min`, `gap`) si usano come target aggiuntivi: lo stimatore di energia (sul ramo conformazione) predice `energy`, lo stimatore di energia minima (sul ramo sequenza) predice `e_min` e il classificatore predice anche `gap`.


**Cosa si intende per MSE e BCE?** Sono le due funzioni di loss, cioè i modi di misurare quanto una predizione è lontana dal target.

- MSE (Mean Squared Error, errore quadratico medio)

- BCE (Binary Cross-Entropy, entropia incrociata binaria)



**Gli stimatori di energia, energia minima e il classificatore finale sono delle MLP**

### Architettura delle reti neurali utilizzate

#### Il mattone comune: il neurone

Un **neurone** riceve dei numeri in ingresso, li moltiplica ciascuno per un **peso**, li somma, aggiunge un **bias** e applica una funzione non lineare (per esempio ReLU):

```
x₁ ─w₁─┐
x₂ ─w₂─┼─► Σ + b ─► ReLU ─► uscita
x₃ ─w₃─┘
```

Un **layer** è un gruppo di neuroni che lavorano in parallelo sugli stessi ingressi. Una **rete** è una serie di layer:

- **input**: i dati;
- **layer nascosti** (*hidden*): i calcoli intermedi;
- **output**: la predizione.

I pesi sono i numeri che il training aggiusta. Le tre reti differiscono per **la forma dell'input** e per **quali ingressi vede ogni neurone**.

**I layer per si possono distinguere da rete neurale a rete neurale poichè puù cambiare come i neuroni sono organizzati, e ci sono operazioni che non sono neuroni.**

#### MLP: tutti collegati a tutti

```
input        hidden       output
 ●  ─────╲╱─── ●
 ●  ──────╳─── ●  ─────►  ●
 ●  ─────╱╲─── ●
```

- **Input**: un vettore di lunghezza fissa, per esempio `z_conf` di 64 numeri.
- **Layer**: ogni neurone è collegato a **tutti** i neuroni del layer precedente (*fully connected*).
- **Output**: un vettore, per esempio 1 numero (`Ê`).
- **Limite**: non sa nulla della struttura dei dati, cioè di chi è vicino a chi.

#### CNN: una finestra che scorre

```
sequenza:  H P P H H P H ...
           └─┬─┘
           filtro (3 pesi) ──► 1 numero
             └─┬─┘
               lo stesso filtro, spostato di 1 ──► 1 numero ...
```

- **Input**: una griglia regolare, cioè un'immagine (2D) o una sequenza (1D) come la stringa HP.
- **Layer**: un blocco che contiene:
  1. **convoluzione** (neuroni): un insieme di **filtri**, ognuno una piccola finestra di pesi (per esempio larga 3) che scorre su tutto l’input. Ogni applicazione del filtro in una posizione è un neurone: vede solo la sua finestra, e tutti i neuroni dello stesso filtro hanno **gli stessi pesi**. Ogni filtro produce una nuova sequenza (o immagine) di valori;
  2. **non linearità** (per esempio ReLU), applicata a ogni valore;
  3. operazioni fisse, **senza pesi**, spesso presenti: **normalizzazione** (riporta i valori a una scala comune) e **pooling** (riduce la griglia prendendo il massimo o la media di gruppi di valori vicini).
- **Più layer**: ogni layer allarga la zona che un neurone "vede". Al primo layer vede 3 residui, al secondo 5, e così via.
- **Output**: di solito il risultato viene riassunto in un vettore (pooling) e passato a un MLP.
- **Nel progetto**: il ramo sequenza può essere una CNN 1D sulla stringa HP.

#### GNN: i vicini sono quelli del grafo

```
     ●───●            ogni nodo raccoglie i messaggi
     │   ┊            dai suoi vicini e si aggiorna
     ●───●───●
```

- **Input**: un grafo, cioè nodi con un vettore di feature ciascuno (H/P, parità…) e archi (legami, contatti).
- **Layer**: un blocco di **message passing** in tre passi:
  1. **messaggio** (neuroni): un piccolo MLP calcola, per ogni arco, cosa il vicino manda al nodo, usando la feature del vicino e il tipo di arco;
  2. **aggregazione** (senza pesi): ogni nodo somma, o fa la media, dei messaggi ricevuti;
  3. **aggiornamento** (neuroni): un altro piccolo MLP combina il vecchio vettore del nodo con la somma dei messaggi e dà il nuovo vettore.

  Gli MLP del messaggio e dell’aggiornamento hanno **gli stessi pesi** per tutti i nodi e tutti gli archi. Spesso si aggiunge anche una **normalizzazione**.
- **Più layer**: dopo k layer un nodo ha informazioni sui nodi fino a k archi di distanza.
- **Output**: un vettore per ogni nodo. Il pooling li riassume in un vettore unico (`z_conf`), poi passato a un MLP.
- **Nel progetto**: il ramo conformazione.

#### In sintesi

|                       | MLP                       | CNN                                  | GNN                |
|:--------------------- | ------------------------- | ------------------------------------ | ------------------ |
| **Input**             | vettore                   | griglia (sequenza, immagine)         | grafo              |
| **Ogni neurone vede** | tutto il layer prima      | una finestra di vicini sulla griglia | i vicini nel grafo |
| **Pesi condivisi**    | no                        | sì, tra le posizioni                 | sì, tra i nodi     |
| **Nel discriminator** | stimatori, classificatore | ramo sequenza                        | ramo conformazione |

In pratica una CNN è come una GNN su un grafo molto regolare, in cui ogni elemento è collegato ai suoi vicini sulla griglia. La GNN permette invece vicini qualsiasi, come i contatti di una proteina ripiegata.

## GNN

Una GNN (*graph neural network*) è una rete neurale che lavora su un grafo, cioè su nodi collegati da archi. Qui il grafo descrive la conformazione: chi tocca chi.

### Il grafo dei contatti

Una conformazione si trasforma in un grafo così:

- **nodi**: i residui, 16 nel nostro caso;

- **archi**: le coppie di residui vicini sul reticolo. Sono di due tipi:

  - **legame**: residui consecutivi nella catena, (i, i+1), sempre presenti;
  - **contatto**: residui vicini sul reticolo ma non consecutivi. Dipendono dalla forma.

**Esempio** con 6 residui, sequenza `H P H P H H`:

```
coordinate:  0:(0,0) 1:(1,0) 2:(2,0) 3:(2,1) 4:(1,1) 5:(0,1)

sul reticolo:            grafo:

  5 ─ 4 ─ 3               legami:   0-1, 1-2, 2-3, 3-4, 4-5
  ┊   ┊   │               contatti: 0-5 (H-H), 1-4 (P-H)
  0 ─ 1 ─ 2
```

(─ e │ sono i legami, ┊ i contatti.) Solo il contatto 0-5 è H-H, quindi l'energia è −1.

Ogni nodo porta un piccolo vettore di **feature**:

| feature   | residuo 0 | residuo 1 |
| --------- | --------- | --------- |
| H?        | 1         | 0         |
| parità    | 0 (pari)  | 1         |
| posizione | 0/15      | 1/15      |
| estremo?  | 1         | 0         |

Il grafo conserva solo **chi tocca chi**. Se la catena viene traslata, ruotata o specchiata, il grafo resta identico. Per questo il modello è invariante per costruzione e non serve l'augmentation dei turni.

### Come lavora una GNN: il *message passing*

Ogni layer della GNN fa la stessa cosa su tutti i nodi:

2. ogni nodo **riceve un messaggio** da ciascun vicino: una funzione della feature del vicino e del tipo di arco;
3. **somma** i messaggi ricevuti;
4. **aggiorna** il proprio vettore combinando quello vecchio con la somma. Questo passo lo fa un piccolo MLP con pesi imparati, condivisi da tutti i nodi.

```
h_i (nuovo) = MLP( h_i ,  Σ_{j vicino di i} messaggio(h_j, tipo arco i-j) )
```

**Perché si adatta così bene al modello HP.** Supponiamo che la rete impari questo messaggio: "se l'arco è un contatto e il vicino è H, manda 1". Dopo un layer, ogni residuo H sa quanti contatti H-H ha. Sommando su tutti i residui H e dividendo per 2 si ottiene **esattamente −E**. Nell'esempio, i residui 0 e 5 ricevono un 1 ciascuno: (1+1)/2 = 1 contatto, cioè E = −1.

L'energia, che per un modello sui turni o sulle coordinate è difficile da ricostruire, qui è a un solo layer di distanza.

**Perché servono più layer (3–4).** Dopo k layer un nodo "vede" i residui fino a k archi di distanza. Questo permette di cogliere più dell'energia, per esempio:

- una zona compatta con un H isolato che potrebbe chiudere un contatto;
- un tratto della catena piegato male accanto a una parte già "nativa".

Sono proprio i difetti tipici dei decoy `near`.

**Pooling.** Alla fine i 16 vettori dei nodi si riassumono in **un solo vettore** `z_conf`, con somma e media. È la "descrizione" della conformazione.
