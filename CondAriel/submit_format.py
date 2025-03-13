import pandas as pd
import h5py
from tqdm import tqdm
import numpy as np 
import os

def get_unique_filename(base_name, extension='.hdf5',directory="./Posterior_Plots"): 
    name_without_ext = os.path.splitext(base_name)[0]
    counter = 1
    while True:
        file_name = f"{name_without_ext}_{counter}{extension}"
        unique_path = os.path.join(directory, file_name)
        if not os.path.exists(unique_path):
            return unique_path
        counter += 1

def get_unique_output(base_name, extension='.hdf5',directory="./output"): 
    name_without_ext = os.path.splitext(base_name)[0]
    counter = 1
    while True:
        file_name = f"{name_without_ext}_{counter}{extension}"
        unique_path = os.path.join(directory, file_name)
        if not os.path.exists(unique_path):
            return unique_path
        counter += 1

def get_unique_graph(base_name, extension='.hdf5',directory="./Grafici_Clustering"): 
    name_without_ext = os.path.splitext(base_name)[0]
    counter = 1
    while True:
        file_name = f"{name_without_ext}_{counter}{extension}"
        unique_path = os.path.join(directory, file_name)
        if not os.path.exists(unique_path):
            return unique_path
        counter += 1
'''
def to_competition_format(tracedata_arr, weights_arr, name="submission.hdf5"):
    """convert input into competition format.
    we assume the test data is arranged in assending order of the planet ID.
    Args:
        tracedata_arr (array): Tracedata array, usually in the form of N x M x 7, where M is the number of tracedata, here we assume tracedata is of equal size.
        It does not have to be but you will need to craete an alternative function if the size is different. 
        weights_arr (array): Weights array, usually in the form of N x M, here we assumed the number of weights is of equal size, it should have the same size as the tracedata
    Returns:
        None
    """
    submit_file = get_unique_filename(name,'.hdf5','./output')
    # submit_file = get_unique_filename(name,extension='.hdf5',directory='./')
    RT_submission = h5py.File(submit_file,'w')
    for n in range(len(tracedata_arr)):
        ## sanity check - samples count should be the same for both
        assert len(tracedata_arr[n]) == len(weights_arr[n])
        ## sanity check - weights must be able to sum to one.
        assert np.isclose(np.sum(weights_arr[n]),1)
        grp = RT_submission.create_group(f"Planet_public{n+1}")
        pl_id = grp.attrs['ID'] = n 
        tracedata = grp.create_dataset('tracedata',data=tracedata_arr[n])         
        weight_adjusted = weights_arr[n]

        weights = grp.create_dataset('weights',data=weight_adjusted)
    RT_submission.close()
'''
def to_competition_format(tracedata_arr, weights_arr, name="submission.hdf5"):
    """Convert input into competition format.
    
    Ci si aspetta che tracedata_arr sia una lista di array (o un array 2D) in 
    cui per ogni elemento la lunghezza sia uguale al numero di pesi corrispondenti.
    In questo caso, se weights_arr[n] ha shape (1,) invece che quella di tracedata_arr[n],
    il vettore verrà replicato. Se la somma dei pesi non è 1, verrà normalizzata.
    """
    submit_file = get_unique_filename(name, '.hdf5', './output')
    RT_submission = h5py.File(submit_file, 'w')
    for n in range(len(tracedata_arr)):
        data = tracedata_arr[n]
        w = weights_arr[n]
        if len(w) != len(data):
            if w.shape[0] == 1:
                w = np.repeat(w, len(data), axis=0)
            else:
                raise AssertionError("Dimension mismatch between tracedata and weights")
        # Normalizza automaticamente i pesi se non sommano ad 1
        total = np.sum(w)
        if not np.isclose(total, 1):
            w = w / total
        grp = RT_submission.create_group(f"Planet_public{n+1}")
        grp.attrs['ID'] = n 
        grp.create_dataset('tracedata', data=data)         
        grp.create_dataset('weights', data=w)
    RT_submission.close()