import numpy as np

import numpy as np

import numpy as np

def check_parameters_valid(params_tuple):
    """
    Checks if the parameters in a tuple are valid float numbers 
    (not NaN or infinite) and not none.

    Args:
        params_tuple: A tuple containing the parameters.

    Returns:
        bool: True if all parameters are valid floats, False otherwise.
    """
    if params_tuple is None:
        return False
    for params in params_tuple:
        if isinstance(params, np.ndarray):
            if  np.any(np.isnan(params)) or np.any(np.isinf(params)):
              return False
            
        elif isinstance(params, (float,int)):
             if np.isnan(params) or np.isinf(params):
                 return False
        elif params is None:
          return False
    return True

def to_observed_matrix(data_file,aux_file):
    # careful, orders in data files are scambled. We need to "align them with id from aux file"
    num = len(data_file.keys())
    id_order = aux_file['planet_ID'].to_numpy()
    observed_spectrum = np.zeros((num,52,4))

    for idx, x in enumerate(id_order):
        current_planet_id = f'Planet_{x}'
        instrument_wlgrid = data_file[current_planet_id]['instrument_wlgrid'][:]
        instrument_spectrum = data_file[current_planet_id]['instrument_spectrum'][:]
        instrument_noise = data_file[current_planet_id]['instrument_noise'][:]
        instrument_wlwidth = data_file[current_planet_id]['instrument_width'][:]
        observed_spectrum[idx,:,:] = np.concatenate([instrument_wlgrid[...,np.newaxis],
                                            instrument_spectrum[...,np.newaxis],
                                            instrument_noise[...,np.newaxis],
                                            instrument_wlwidth[...,np.newaxis]],axis=-1)
    return observed_spectrum


def standardise(arr, mean, std):
    return (arr-mean)/std

def transform_back(arr, mean, std):
    return arr*std+mean

def augment_data(arr, noise, repeat=10):
    noise_profile = np.random.normal(loc=0, scale=noise, size=(repeat,arr.shape[0], arr.shape[1]))
    ## produce noised version of the spectra
    aug_arr = arr[np.newaxis, ...] + noise_profile
    return aug_arr

def visualise_spectrum(spectrum):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10,6))
    plt.errorbar(x=spectrum[:,0], y= spectrum[:,1], yerr=spectrum[:,2] )
    ## usually we visualise it in log-scale
    plt.xscale('log')
    plt.show()
    
import numpy as np

def transform_and_reshape(y_pred, mean, std, instances, N_testdata):
    """
    Trasforma i dati predetti alla scala originale.

    Args:
        y_pred (np.array): Dati predetti da trasformare e rimodellare.
        mean (np.array): Valori medi originali.
        std (np.array): Deviazioni standard originali.
        instances (int): Numero di istanze.
        N_testdata (int): Lunghezza dei dati di test.

    Returns:
        np.array: Dati trasformati e rimodellati.
    """
    # Ensure mean and std are numpy arrays
    mean = np.array(mean.values if hasattr(mean, 'values') else mean)
    std = np.array(std.values if hasattr(std, 'values') else std)
    
    # Trasformazione e reshaping
    y_pred_transformed = y_pred * std + mean
    y_pred_reshaped = y_pred_transformed.reshape(instances, N_testdata, -1)
    return y_pred_reshaped