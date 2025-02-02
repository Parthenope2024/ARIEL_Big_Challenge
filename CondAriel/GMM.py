import numpy as np
import pandas as pd
import os
import h5py
import random
import sys
import corner
import matplotlib.pyplot as plt
import taurex.log

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn import mixture
from bayesian_bootstrap import bayesian_bootstrap
from helper import *
from helper import *
from preprocessing import *
from submit_format import to_competition_format
from posterior_utils import *
from spectral_metric import *
from FM_utils_final import *

taurex.log.disableLogging()
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize)

n_repeat = 5
random_state = 420

# Lettura dati dai file 
aux = np.load('aux.npy')
spec_matrix = np.load('spectra.npy')
noise = np.load('noise.npy')
labels = np.load('label.npy')
validTraces = np.load('validTraces.npy')

# Setting dei path

## PARTIAL DATASET
# training_path = './Training/'

## FULL DATASET
training_path = './Full_Dataset/Level2Data/'
training_GT_path = os.path.join(training_path, 'Ground Truth Package')
trace_GT = h5py.File(os.path.join(training_GT_path, 'TraceData.hdf5'), "r")

validTraces = validTraces.astype(int)
num_spectra = spec_matrix.shape[0]

vt = validTraces
test_ind = np.sort(vt - 1)  # vt[:split]-1
train_ind = np.setdiff1d(np.arange(num_spectra), test_ind)
plot_ind = random.sample(range(len(test_ind)), 10)
spectra_ind = random.sample(range(len(test_ind)), 10)

test_spectra = spec_matrix[test_ind, :]
test_spectra = augment_data(test_spectra, noise[test_ind, :], repeat=1)
test_spectra = test_spectra.reshape(-1, spec_matrix.shape[1])
# test_spectra = normalize(test_spectra, axis=0, norm='max')

train_spectra = spec_matrix[train_ind, :]
train_spectra = augment_data(train_spectra, noise[train_ind, :], repeat=n_repeat)
train_spectra = train_spectra.reshape(-1, spec_matrix.shape[1])
# train_spectra = normalize(train_spectra, axis=0, norm='max')

train_aux = aux[train_ind, :]
train_aux = np.repeat(train_aux, repeats=n_repeat, axis=0)
test_aux = aux[test_ind, :]

train_labels = labels[train_ind, :]
train_labels = np.repeat(train_labels, repeats=n_repeat, axis=0)
test_labels = labels[test_ind, :]

# Setting dei parametri cercati
labels_names = ['planet_radius', 'planet_temp', 'log_H2O', 'log_CO2', 'log_CO', 'log_CH4', 'log_NH3']
# Controllo della compatibilita' dei size con l'input richiesto dal modello
RJUP = 69911000
MJUP = 1.898e27
RSOL = 696340000
Rs = aux[:, 2] / RSOL  # ['star_radius_m']
# Rp = aux_df['planet_radius_m']/RJUP
Mp = aux[:, 4] / MJUP  # ['planet_mass_kg']
n_samples = 1000

K1 = 10
K2 = 20
GMM_i = []
Labels_i = []

# Primo clustering con GaussianMixture
gmm = mixture.GaussianMixture(n_components=K1, random_state=random_state, max_iter=500).fit(train_aux)
labels_1 = gmm.predict(train_aux)
for i in range(K1):
    spectra_i = np.where(labels_1 == i)[0]  # Indici degli spettri che appartengono al cluster i
    print("Clustering spettro #", i, " -> ", len(spectra_i))
    
    # Secondo clustering (2° livello) sul sottoinsieme di spettri
    tmp = mixture.GaussianMixture(n_components=K2, random_state=random_state, max_iter=500).fit(train_spectra[spectra_i, :])
    labels_2 = tmp.predict(train_spectra[spectra_i, :])
    GMM_i.append(tmp)
    Labels_i.append(labels_2)
    
    for j in range(K2):
        spectra_j = np.where(labels_2 == j)[0]
        if len(spectra_j) == 1:
            # Se il j-simo sottocluster dell'i-simo cluster contiene un solo spettro, lo consideriamo outlier
            print(f"\tSpettri nel cluster # {i} : {j}, -> len:{len(spectra_j)} [OUTLIER]")

posterior_scores = []
spectral_scores = []
bounds_matrix = default_prior_bounds()
beta = 0.8
q_list = np.linspace(0.01, 0.99, 10)
## Path variables
opacity_path = "./XSEC/"
CIA_path = "./HITRAN"
## read in spectral grid
ariel_wlgrid, ariel_wlwidth, ariel_ngrid, ariel_wnwidth = ariel_resolution()
## Initialise base T3 model for ADC2023
fm = initialise_forward_model(opacity_path, CIA_path)

# Ciclo sui test sample
for X in range(len(test_ind)):
    min_w = 1e-8
    idx1 = gmm.predict(test_aux[X, :].reshape(1, -1))[0]
    km = GMM_i[idx1]
    labels_2 = Labels_i[idx1]
    idx2 = km.predict(test_spectra[X, :].reshape(1, -1))[0]
    idx_1 = np.where(labels_1 == idx1)[0]
    idx_2 = np.where(labels_2 == idx2)[0]
    lab = train_labels[idx_1[idx_2], :6]
    posterior = lab
    weights1 = np.ones((posterior.shape[0], 1)) / np.sum(np.ones(posterior.shape[0]))

    try:
        tr_GT = trace_GT[f'Planet_{test_ind[X]+1}']['tracedata'][()]
    except KeyError:
        print(f"Skipping test sample {X} due to KeyError when accessing trace data.")
        continue

    # Se tr_GT è una tuple, estraiamo il primo elemento
    if isinstance(tr_GT, tuple):
        if not tr_GT:
            print(f"Skipping test sample {X} because tr_GT is an empty tuple.")
            continue
        tr_GT = tr_GT[0]

    # Verifica che tr_GT sia un array valido
    if not isinstance(tr_GT, np.ndarray):
        print(f"Skipping test sample {X} because tr_GT is not a valid array.")
        continue

    try:
        wh_GT = trace_GT[f'Planet_{test_ind[X]+1}']['weights'][()]
    except KeyError:
        print(f"Skipping test sample {X} due to KeyError when accessing weight data.")
        continue
    except Exception:
        print(f"Skipping test sample {X} due to an unexpected error when accessing weight data.")
        continue

    if posterior.shape[0] < 2:
        print(f"Skipping test sample {X} because of too few traces. Dimension is {posterior.shape[0]}")
        continue

    if np.isnan(tr_GT).sum() > 0 or np.isinf(tr_GT).sum() > 0:
        continue
    if np.isnan(weights1).sum() > 0 or np.isinf(weights1).sum() > 0:
        continue

    try:
        score = compute_posterior_loss(posterior, weights1, tr_GT, wh_GT, bounds_matrix)
    except (IndexError, FloatingPointError) as e:
        continue
    else:
        if not np.isnan(score):
            posterior_scores.append(score)

    if X in spectra_ind:
        proxy_compute_spectrum = setup_dedicated_fm(fm, X, Rs, Mp, ariel_ngrid, ariel_wnwidth)
        score = compute_spectral_loss(posterior, weights1, tr_GT, wh_GT, bounds_matrix, proxy_compute_spectrum, q_list)
        spectral_scores.append(score)

avg_posterior_score = np.mean(posterior_scores)
print(f'Posterior_Score: {avg_posterior_score}')

avg_spectral_score = np.mean(spectral_scores)
print(f'Spectral_Score: {avg_spectral_score}')

final_score = (1 - beta) * avg_spectral_score + beta * avg_posterior_score
print(f"final loss is {final_score:.4f}")

# Ciclo per generare i plot
for X in plot_ind:
    idx1 = gmm.predict(test_aux[X, :].reshape(1, -1))[0]
    km = GMM_i[idx1]
    labels_2 = Labels_i[idx1]
    idx2 = km.predict(test_spectra[X, :].reshape(1, -1))[0]
    score = km.score_samples(spec_matrix[X, :].reshape(1, -1))[0]
    idx_1 = np.where(labels_1 == idx1)[0]
    idx_2 = np.where(labels_2 == idx2)[0]
    lab = train_labels[idx_1[idx_2], :]
    mean_lab = np.mean(lab, axis=0)
    
    try:
        tr_GT = trace_GT[f'Planet_train{test_ind[X]+1}']['tracedata'][()]
    except KeyError:
        print(f"Skipping plot {X} due to KeyError when accessing trace data.")
        continue

    if isinstance(tr_GT, tuple):
        if not tr_GT:
            print(f"Skipping plot {X} because tr_GT is an empty tuple.")
            continue
        tr_GT = tr_GT[0]

    if not isinstance(tr_GT, np.ndarray):
        print(f"Skipping plot {X} because tr_GT is not a valid array.")
        continue

    try:
        wh_GT = trace_GT[f'Planet_train{test_ind[X]+1}']['weights'][()]
    except KeyError:
        print(f"Skipping plot {X} due to KeyError when accessing weight data.")
        continue

    # Se tr_GT ha più di un elemento, selezioniamo il secondo, altrimenti il primo
    if tr_GT.shape[0] > 1:
        Tp = tr_GT[1]
    else:
        Tp = tr_GT[0]

    # Calcola il punteggio e aggiungi ai risultati
    score = compute_posterior_loss(posterior, weights1, tr_GT, wh_GT, bounds_matrix)
    posterior_scores.append(score)
    if X in spectra_ind:
        proxy_compute_spectrum = setup_dedicated_fm(fm, X, Rs, Mp, ariel_ngrid, ariel_wnwidth)
        score = compute_spectral_loss(posterior, weights1, tr_GT, wh_GT, bounds_matrix, proxy_compute_spectrum, q_list)
        spectral_scores.append(score)
    # Generazione dei corner plot
    figure = corner.corner(tr_GT, quiet=True)
    axes = np.array(figure.axes).reshape((tr_GT.shape[1], tr_GT.shape[1]))
    for i in range(tr_GT.shape[1]):
        ax = axes[i, i]
        ax.sharex(axes[tr_GT.shape[1]-1, i])
        ax.axvline(test_labels[X, i], color="g")
        ax.axvline(mean_lab[i], color="r")
        ax.relim()
        ax.autoscale()
        ax.set_title(labels_names[i])
    figure.set_figheight(8.5)
    figure.set_figwidth(12)
    corner.corner(lab, fig=figure, quiet=True, color='red')
    figure.savefig(f'./GMM_plots/corner_plot_{X}.png')
plt.show()