# Descrizione : 

- [Descrizione Repository](#Descrizione)
- [Prima Parte : MCDropout](#PrimaParte)
- [Seconda Parte : GMM](#SecondaParte)
 
---

# Descrizione : 
## Contenuto Repository

La repository contiene varie cartelle:
- GMM_plots : Plot dell'algoritmo con doppio clustering
- outputs : risultati dell'algoritmo (.hdf5)
- Clusters : Dimensioni e distribuzioni dei cluster e dei risultati del clustering (pianeti categorizzati)
- Grafici Clustering : Grafici dell'andamento degli score risultanti dal clustering
- `GaussianMixtureModel.ipynb` : Notebook principale per l'esecuzione dell'algoritmo
- `GUI.py` : Interfaccia grafica per la semplificazione visiva dei risultati
- `outputs.csv` : File contenente i risultati, base per la generazione dei grafici
  
---

# PrimaParte :
> Clustering *MonteCarlo Dropout*

Questa parte e' la base del lavoro di tesi, usando la `base_pipeline` di ARIEL.
Si utilizza un modello di *MonteCarlo Dropout* per le predizioni, ed effettuiamo le validazioni con `tqdm` e:

`y_pred_valid = model([std_valid_spectra,std_valid_Rs],training=True)`
`y_valid_distribution[i] += y_pred_valid`

Questo modello e' la base del lavoro di tesi, lavorando unicamente con gli spettri.

---

# SecondaParte:
> Clustering *Gaussian Mixture Model*

Questa parte e' la principale del lavoro di tesi, e utilizza il `Gaussian Mixture Model` per eseguire un **doppio clustering**, prima sui dati *ausiliari* (raggio, temperatura, massa,..), e successivamente sugli spettri.
L'algoritmo utilizza la coppia (K1,K2), per eseguire i due clustering, ricavata mediante `Bayesian Information Criterion`, che ritorna il modello con minimo B.I.C. tra tutti i modelli disponibili, e nel nostro caso utilizza (19,15) come coppia.

I dati sono preprocessati e aumentati con il rumore sintetico fornito da ARIEL, mediante `augment_data()`, che aumenta gli spettri con il rumore di ARIELRad, e successivamente gli applica una normalizzazione e scalatura con `StandardScaler()`, per avere dati con **media nulla** e **devianza standard** unitaria, per *avere dati sulla stessa scala*, cosi' che nessuna componente sovrasti le altre.

L'algoritmo genera gli score tramite test K-S (posterior) e Huber Loss (spectral), calcola il valore medio e lo ritorna come score, e successivamente calcola il `final_score` come combinazione pesata dei due score
> final_score = 0.8 * posterior_score + 0.2 * spectral_score



![Auto Assign](https://github.com/Parthenope2024/demo-repository/actions/workflows/auto-assign.yml/badge.svg)
![Proof HTML](https://github.com/Parthenope2024/demo-repository/actions/workflows/proof-html.yml/badge.svg)

![Auto Assign](https://github.com/Parthenope2024/demo-repository/actions/workflows/auto-assign.yml/badge.svg)
![Proof HTML](https://github.com/Parthenope2024/demo-repository/actions/workflows/proof-html.yml/badge.svg)
