# Descrizione : 

---

# Prima Parte:

## 1) Contenuto Repository:

- Codice : Il codice è contenuto principalmente nel file 'SpectralData-works.ipynb', file **Notebook Jupyter**, le cui "librerie" sono nei vari file '.py' all'interno della cartella (attualmente compresso in un '.7z'.
- Archivi : Gli archivi 'Rayleigh', 'XSEC' e 'CIA' sono archivi di file per l'utilizzo sugli spettri
- Cartelle:
  Le cartelle nella repo sono così composte:
    - OUTPUT_TEST : Cartella per descrivere l'andamento degli output *in base ai sample/indici utilizzati* (fare riferimento all'issue > Problema Dati)
    - Posterior_Distributions : Cartella contenente i plot delle distribuzioni Bayesiane a posteriori, divise per **aggregate** (tutti i parametri) e **single_plots** (singolo parametro)
    - best_run : cartella (il cui contenuto può variare) contenente i dati riguardo la migliore run fatta fin'ora
    - outputs : cartella contenente *i file 'submission_<counter>.hdf5'*

---

## 2) Struttura '.ipynb':

Il file 'SpectralData-works.ipynb' contiene il codice principale per la run, insieme ai commenti ed alle spiegazioni riguardo il codice e la spiegazione.

Tutto ciò che serve per eseguire il codice e farlo funzionare correttamente è indicato **all'interno del file**, ma ci sono dei requisiti di installazione per eseguirlo, e si trovano nel file 'requirements.txt'. Basterà quindi **installare *tutti* i pacchetti presenti nel 'txt'**.

Una volta installati, bisognerà assicurarsi di avere nella cartella del progetto le cartelle 'Training/', 'Test/', 'xsec/', 'Rayleigh/', 'CIA/' , 'outputs/', 'Posterior_Distributions/single_plots/', 'Posterior_Distributions/aggregate/', affinchè non vi siano errori. Tutte queste cartelle sono all'interno della repository, alcune **come archivi**, altre *come directory vere e proprie*.

---

## 3) Compilazione ed Output:

Una volta eseguito il codice 
  >potrebbe impiegarci decine di minuti

si vedranno gli output nelle rispettive cartelle, e sarà possibile ottenere un quadro generale del funzionamento e della bontà dell'algoritmo.

Per avere un'idea visiva degli output, si cerca di avere la distribuzione *predetta* dei parametri **molto vicina al valore reale del parametro**, in modo da capire subito se l'algoritmo è buono o no, in base ai parametri di 'loss' e 'validation_loss'. 

Essi sono visibili durante l'esecuzione dell'algoritmo, venendo stampati a schermo sul Notebook Jupyter durante l'esecuzione. In questo modo è possibile avere un doppio confronto sulla bontà del modello e dei valori dei dati risultanti.

**Si presti attenzione al fatto che, essendo il DataSet fornito direttamente da ARIEL (quindi poco modificabile), ed il fatto che vi è un problema noto sui dati** 
  >fare riferimento all'issue/discussione [#https://github.com/orgs/Parthenope2024/discussions/4]

** si potrebbero avere output di 'Not-a-Number' o 'infinito', ma ciò non riguarda unicamente il modello**, ma bensi' *il dataset fornito da ARIEL*

---

## GUI Update!

Ho creato una piccola Graphic User Interface per visualizzare l'output del modello.
Il file 'GraphicInterface.py' esegue il Notebook, salva l'output in un file chiamato 'SpectralOutput.ipynb', e legge il contenuto degli output cui siamo interessati da un .csv.
Successivamente, crea una piccola GUI per aiutarci a visualizzare i dati di output a schermo.

---

# Seconda Parte ('Gaussian Mixture Model'):

## 1) Codice:
Il codice e' nel file 'GMM.ipynb', visualizzabile anche mediante il file 'GMM.py'. I dati (di entrambi i file) vengono presi dai file '.npy', che verranno estratti direttamente nel codice, mediante apposite librerie

## 2) Differenze con il codice originale:
La principale differenza e' la **complessita' d'esecuzione**, che in questo file e' molto maggiore, dovendo eseguire un algoritmo *molto piu' pesante* rispetto ad un classico 'KMeans', difatti runna il ***Gaussian Mixture Model***, un algoritmo di 'Soft Clustering', che a differenza dell' 'Hard Clustering', un dato puo' appartenere **a piu' cluster**, in base ad un grado.

## 3) Compilazione ed output:
La compilazione e l'output avvengono allo stesso modo del vecchio file, ma la differenza e' che si avra' un output dopo piu' tempo e si vedranno dei plot delle distribuzioni bayesiane molto piu' complesse, in riferimento a piu' dati ecc.

![Auto Assign](https://github.com/Parthenope2024/demo-repository/actions/workflows/auto-assign.yml/badge.svg)
![Proof HTML](https://github.com/Parthenope2024/demo-repository/actions/workflows/proof-html.yml/badge.svg)
