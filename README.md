# Descrizione : 

---

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

**, si potrebbero avere output di 'Not-a-Number' o 'infinito', ma ciò non riguarda unicamente il modello**

![Auto Assign](https://github.com/Parthenope2024/demo-repository/actions/workflows/auto-assign.yml/badge.svg)
![Proof HTML](https://github.com/Parthenope2024/demo-repository/actions/workflows/proof-html.yml/badge.svg)
