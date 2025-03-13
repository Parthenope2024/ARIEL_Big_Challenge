from functools import partial
import nestle
import numpy as np
import taurex
def lerp(v0, v1, t):
    """Linearly interpolate between v0 and v1 by a factor of t."""
    return (1 - t) * v0 + t * v1

def quantile_index(n, q, alpha=1, beta=1):
    """Compute the index of the quantile q in a sample of size n."""
    return q * (n - alpha - beta + 1) + alpha - 1

def loop_func(l, f, func, samples):
    interpolate = f != 0
    if not interpolate:
        return func(idx=l, trace=samples)
    else:
        r = l + 1
        v0 = func(idx=l, trace=samples)
        v1 = func(idx=r, trace=samples)
        return lerp(v0,v1, f)

def compute_quantile_indices(total_examples, quantiles):
    """Compute the indices of the quantiles in a sample of size n."""
    import numpy as np
    quantiles_index = np.array([quantile_index(total_examples,q) for q in quantiles])
    
    whole_indices = quantiles_index.astype(np.int64)
    leftovers = quantiles_index - whole_indices
    
    return whole_indices, leftovers


def compute_sample_quantiles(func, samples, quantiles):
    from tqdm import tqdm
    from joblib import Parallel, delayed


    """Compute the quantiles of a sample."""
    total_samples = samples.shape[0]
    left_index, fractional = compute_quantile_indices(total_samples, quantiles)
    partial_loop_func = partial(loop_func, func = func, samples = samples)
    quantile_result = [partial_loop_func(l,f) for l,f in zip(left_index, fractional)]

    return quantile_result


def compute_approx_mean_and_bound(tr, weights, proxy_compute_spectrum, quantiles):
    tr_resampled = nestle.resample_equal(tr, weights)
    tr_resampled = np.array(tr_resampled, dtype=np.float64)
    weights_sorted_idx = weights.argsort()
    weights_sorted_idx = weights_sorted_idx.astype(np.int64)
    tr_sorted = tr_resampled[weights_sorted_idx]
    qs = compute_sample_quantiles(proxy_compute_spectrum, tr_sorted, quantiles)
    q1, q2, q3 = np.quantile(np.array(qs), q=(0.25, 0.5, 0.75), axis=0)
    return q1, q2, q3


def compute_ariel_spectrum(idx, trace, fm, Rs, Mp , ariel_wngrid, ariel_wnwidth):
    """
    Compute spectra from an instance of the (resampled) tracedata in ariel resolution
    """
    if(idx>=len(trace)):
        raise IndexError(f"Index out of range")
    
    def get_value(index, default):
        # Check if trace[idx] exists and has the index
        if len(trace)>0 and trace[idx].shape[0] > index:
           return trace[idx][index]
        else:
            return default

    Tp = get_value(1, 1000) #default temperature
    Rp = get_value(0, 1.0)  # Default radius
    if isinstance(Rp, np.ndarray) and Rp.size > 1:
        print(f"Warning: more than 1 Rp value detected in the trace! Using first value")
        Rp=Rp[0]
    if isinstance(Rp, np.ndarray):
        Rp = Rp.item()
    
    Mp = get_value(4,1.0)
    if isinstance(Mp, np.ndarray) and Mp.size >1:
       print(f"Warning: more than 1 Mp value detected in the trace! Using first value")
       Mp = Mp[0]
    if isinstance(Mp, np.ndarray):
        Mp = Mp.item()
   
    output, fm = update_forward_model(
        fm,
        Rs=Rs,
        Rp=Rp,
        Mp=Mp,
        Tp=Tp,
        X_h2o = get_value(2, -5),   # Use a default if not available
        X_ch4 = get_value(5, -5),   # Use a default if not available
        X_co = get_value(4, -5),    # Use a default if not available
        X_co2 = get_value(3, -5),  # Use a default if not available
        X_nh3 = get_value(6, -5)   # Use a default if not available
    )
    # bin them down
    if output is not None:
        _, binned_spectrum = bin_forward_model(output[0], output[1], ariel_wngrid, ariel_wnwidth)
    else:
        return None
    return binned_spectrum

def update_forward_model(forward_model, Rs, Rp, Mp, Tp,X_h2o, X_ch4, X_co, X_co2, X_nh3):
    """Generate a corresponding forward model from a list of input parameters"""
    from taurex.constants import RSOL

    forward_model.star._radius = Rs*RSOL
    forward_model.planet.set_planet_radius(Rp)
    forward_model.planet.set_planet_mass(Mp)

    if np.isnan(Tp) or np.isinf(Tp):
        print("Temperature is Nan/Inf", Tp)
        return None, None
    
    if np.isnan(X_h2o) or np.isinf(X_h2o):
        print("H2O is NaN/Inf:", X_h2o)
        return None, None
    if np.isnan(X_ch4) or np.isinf(X_ch4):
        print("CH4 is NaN/Inf:", X_ch4)
        return None, None
    if np.isnan(X_co) or np.isinf(X_co):
        print("CO is NaN/Inf:", X_co)
        return None, None
    if np.isnan(X_co2) or np.isinf(X_co2):
        print("CO2 is NaN/Inf:", X_co2)
        return None, None
    if np.isnan(X_nh3) or np.isinf(X_nh3):
        print("NH3 is NaN/Inf:", X_nh3)
        return None, None

    try:
       forward_model['T'] = Tp
       ## add more parameters
       forward_model['H2O'] = 10**X_h2o
       forward_model['CH4'] = 10**X_ch4
       forward_model['CO2'] = 10**X_co2
       forward_model['CO'] = 10**X_co
       forward_model['NH3'] = 10**X_nh3

       res = forward_model.model()
       return res,forward_model
    except Exception as e:
        print('Error in update_forward_model', e)
        return None, None


def bin_forward_model(native_wl, native_spectrum, wn, wn_width):
    from taurex.binning import FluxBinner

    fb = FluxBinner(wn, wn_width)
    out = fb.bindown(native_wl, native_spectrum)
    return out[0], out[1]


def initialise_forward_model(opacity_path, CIA_path, he_h2=0.17, n_layers=100, P_top=1e-5, P_surface=1e6,):
    """Initialise the official forward model for ADC2023."""
    from taurex.cache import OpacityCache, CIACache
    from taurex.contributions import AbsorptionContribution, RayleighContribution, CIAContribution
    from taurex.data.stellar import BlackbodyStar
    from taurex.data import Planet
    from taurex.data.profiles.chemistry import TaurexChemistry
    from taurex.data.profiles.chemistry import ConstantGas
    from taurex.data.profiles.temperature import Isothermal
    from taurex.model import TransmissionModel
            
    OpacityCache().set_opacity_path(opacity_path)
    CIACache().set_cia_path(CIA_path)

    # Planets
    planet = Planet(planet_radius=1, planet_mass=1,)

    # Stars
    star = BlackbodyStar(radius=1,)

    # Chemistry
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=he_h2)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CH4', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CO2', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CO', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('NH3', mix_ratio=10**-5))

    # Temperature structure
    IsoT = Isothermal(1000)

    # Putting everything together
    forward_model = TransmissionModel(planet=planet, temperature_profile=IsoT, chemistry=chemistry, star=star, atm_min_pressure=P_top, atm_max_pressure=P_surface, nlayers=n_layers)

    # enabling contribution from various opacity sources
    forward_model.add_contribution(AbsorptionContribution())
    forward_model.add_contribution(RayleighContribution())
    forward_model.add_contribution(CIAContribution())

    forward_model.build()
    forward_model.model()
    return forward_model

def setup_dedicated_fm(fm, X, Rs, Mp, ariel_wngrid, ariel_wnwidth):
    def proxy_compute_spectrum(params):
        
        # DEBUG: print input to function
        # print("Input parameters:",params) 
        
        for i in range(len(params)):
          if np.isnan(params[i]) or np.isinf(params[i]):
             print(f"Invalid value in input parameters at index {i}:", params)
             return None # Returning None if inputs have nan

        # Example of checking before log
        try:
          radius =  params[0] 
          if radius <= 0:
            print("Invalid radius parameter ", radius)
            return None
        except Exception as e:
           print(e)
           return None
        try:
            spectrum, path_length = fm.compute_spectrum(params, Rs[X], Mp[X], ariel_wngrid, ariel_wnwidth)
        except Exception as e:
            print('Error calculating spectrum', e)
            return None
        #DEBUG
        # print("path_length values:",path_length)
        for i in range(len(path_length)):
            for j in range(len(path_length[i])):
                if np.isnan(path_length[i][j]) or np.isinf(path_length[i][j]):
                    print(f"Invalid value in path_length at layer {i} and index {j}:", path_length[i][j])
                    return None # Returning None if pathlength has nan
        return spectrum, path_length
    return proxy_compute_spectrum


def ariel_resolution():
    import numpy as np
    from taurex.util.util import wnwidth_to_wlwidth
    
    #griglie di lunghezze d'onda
    wlgrid = np.array([0.55      , 0.7       , 0.95      , 1.156375  , 1.27490344,
       1.40558104, 1.5496531 , 1.70849254, 1.88361302, 1.9695975 ,
       2.00918641, 2.04957106, 2.09076743, 2.13279186, 2.17566098,
       2.21939176, 2.26400154, 2.30950797, 2.35592908, 2.40328325,
       2.45158925, 2.50086619, 2.5511336 , 2.60241139, 2.65471985,
       2.70807972, 2.76251213, 2.81803862, 2.8746812 , 2.93246229,
       2.99140478, 3.05153202, 3.11286781, 3.17543645, 3.23926272,
       3.30437191, 3.37078978, 3.43854266, 3.50765736, 3.57816128,
       3.65008232, 3.72344897, 4.03216667, 4.30545796, 4.59727234,
       4.90886524, 5.24157722, 5.59683967, 5.97618103, 6.3812333 ,
       6.81373911, 7.2755592 ])[::-1]
    wlwidth = np.array([0.10083333, 0.20416667, 0.30767045, 0.11301861, 0.12460302,
       0.13737483, 0.15145575, 0.16697996, 0.18409541, 0.03919888,
       0.03998678, 0.04079051, 0.0416104 , 0.04244677, 0.04329995,
       0.04417028, 0.0450581 , 0.04596377, 0.04688764, 0.04783008,
       0.04879147, 0.04977218, 0.0507726 , 0.05179313, 0.05283417,
       0.05389614, 0.05497945, 0.05608453, 0.05721183, 0.05836179,
       0.05953486, 0.06073151, 0.06195222, 0.06319746, 0.06446773,
       0.06576353, 0.06708537, 0.06843379, 0.06980931, 0.07121248,
       0.07264385, 0.07410399, 0.26461764, 0.28255283, 0.30170364,
       0.32215244, 0.34398722, 0.36730191, 0.39219681, 0.41877904,
       0.44716295, 0.47747067])[::-1]
    wngrid = 10000/wlgrid
    wnwidth = wnwidth_to_wlwidth(wlgrid,wlwidth )
    return wlgrid, wlwidth, wngrid, wnwidth

def preprocessing(input, error_flag= False):
    """Preprocess the input before feeding it into the main computation loop (unused)"""
    return input, error_flag

def check_output(q1, q2, q3, min_val = 1e-8, max_val = 0.1):
    """performs two kinds of check, check for nan , inf and -inf, another checks for large finite values """
    q1 = np.nan_to_num(q1, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q2 = np.nan_to_num(q2, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q3 = np.nan_to_num(q3, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q1 = np.clip(q1, min_val, max_val)
    q2 = np.clip(q2, min_val, max_val)
    q3 = np.clip(q3, min_val, max_val)

    assert np.sum(~np.isfinite(q1)) == 0, "Unphysical values identified"
    assert np.sum(~np.isfinite(q2)) == 0, "Unphysical values identified"
    assert np.sum(~np.isfinite(q3)) == 0, "Unphysical values identified"
    return q1, q2, q3

''' OLD:

from functools import partial
import nestle
import numpy as np

def lerp(v0, v1, t):
    """Linearly interpolate between v0 and v1 by a factor of t."""
    return (1 - t) * v0 + t * v1

def quantile_index(n, q, alpha=1, beta=1):
    """Compute the index of the quantile q in a sample of size n."""
    return q * (n - alpha - beta + 1) + alpha - 1

def loop_func(l, f, func, samples):
    interpolate = f != 0
    if not interpolate:
        return func(idx=l, trace=samples)
    else:
        r = l + 1
        v0 = func(idx=l, trace=samples)
        v1 = func(idx=r, trace=samples)
        return lerp(v0,v1, f)

def compute_quantile_indices(total_examples, quantiles):
    """Compute the indices of the quantiles in a sample of size n."""
    import numpy as np
    quantiles_index = np.array([quantile_index(total_examples,q) for q in quantiles])
    
    whole_indices = quantiles_index.astype(np.int64)
    leftovers = quantiles_index - whole_indices
    
    return whole_indices, leftovers


def compute_sample_quantiles(func, samples, quantiles):
    from tqdm import tqdm
    from joblib import Parallel, delayed


    """Compute the quantiles of a sample."""
    total_samples = samples.shape[0]
    left_index, fractional = compute_quantile_indices(total_samples, quantiles)
    partial_loop_func = partial(loop_func, func = func, samples = samples)
    quantile_result = [partial_loop_func(l,f) for l,f in zip(left_index, fractional)]

    return quantile_result


def compute_approx_mean_and_bound(tr, weights, proxy_compute_spectrum, quantiles):
    tr_resampled = nestle.resample_equal(tr, weights)
    tr_resampled = np.array(tr_resampled, dtype=np.float64)
    weights_sorted_idx = weights.argsort()
    weights_sorted_idx = weights_sorted_idx.astype(np.int64)
    tr_sorted = tr_resampled[weights_sorted_idx]
    qs = compute_sample_quantiles(proxy_compute_spectrum, tr_sorted, quantiles)
    q1, q2, q3 = np.quantile(np.array(qs), q=(0.25, 0.5, 0.75), axis=0)
    return q1, q2, q3

def compute_ariel_spectrum(idx, trace, fm, Rs, Mp , ariel_wngrid, ariel_wnwidth):
    """
    Compute spectra from an instance of the (resampled) tracedata in ariel resolution
    """
    if(idx>=len(trace)):
        raise IndexError(f"Index out of range")
    
    def get_value(index, default):
        # Check if trace[idx] exists and has the index
        if len(trace)>0 and trace[idx].shape[0] > index:
           return trace[idx][index]
        else:
            return default

    Tp = get_value(1, 1000) #default temperature
    Rp = get_value(0, 1.0)  # Default radius
    if isinstance(Rp, np.ndarray) and Rp.size > 1:
        print(f"Warning: more than 1 Rp value detected in the trace! Using first value")
        Rp=Rp[0]
    if isinstance(Rp, np.ndarray):
        Rp = Rp.item()
    
    Mp = get_value(4,1.0)
    if isinstance(Mp, np.ndarray) and Mp.size >1:
       print(f"Warning: more than 1 Mp value detected in the trace! Using first value")
       Mp = Mp[0]
    if isinstance(Mp, np.ndarray):
        Mp = Mp.item()
   
    output, fm = update_forward_model(
        fm,
        Rs=Rs,
        Rp=Rp,
        Mp=Mp,
        Tp=Tp,
        X_h2o = get_value(2, -5),   # Use a default if not available
        X_ch4 = get_value(5, -5),   # Use a default if not available
        X_co = get_value(4, -5),    # Use a default if not available
        X_co2 = get_value(3, -5),  # Use a default if not available
        X_nh3 = get_value(6, -5)   # Use a default if not available
    )
    # bin them down
    _, binned_spectrum = bin_forward_model(output[0], output[1], ariel_wngrid, ariel_wnwidth)
    return binned_spectrum

def update_forward_model(forward_model, Rs, Rp, Mp, Tp,X_h2o, X_ch4, X_co, X_co2, X_nh3):
    """Generate a corresponding forward model from a list of input parameters"""
    from taurex.constants import RSOL

    forward_model.star._radius = Rs*RSOL
    forward_model.planet.set_planet_radius(Rp)
    forward_model.planet.set_planet_mass(Mp)
    
    forward_model['T'] = Tp
    ## add more parameters
    forward_model['H2O'] = 10**X_h2o
    forward_model['CH4'] = 10**X_ch4
    forward_model['CO2'] = 10**X_co2
    forward_model['CO'] = 10**X_co
    forward_model['NH3'] = 10**X_nh3

    res = forward_model.model()
    return res,forward_model


def bin_forward_model(native_wl, native_spectrum, wn, wn_width):
    from taurex.binning import FluxBinner

    fb = FluxBinner(wn, wn_width)

    out = fb.bindown(native_wl, native_spectrum)

    return out[0], out[1]


def initialise_forward_model(opacity_path, CIA_path, he_h2=0.17, n_layers=100, P_top=1e-5, P_surface=1e6,):
    """Initialise the official forward model for ADC2023."""
    from taurex.cache import OpacityCache, CIACache
    from taurex.contributions import AbsorptionContribution, RayleighContribution, CIAContribution
    from taurex.data.stellar import BlackbodyStar
    from taurex.data import Planet
    from taurex.data.profiles.chemistry import TaurexChemistry
    from taurex.data.profiles.chemistry import ConstantGas
    from taurex.data.profiles.temperature import Isothermal
    from taurex.model import TransmissionModel
            
    OpacityCache().set_opacity_path(opacity_path)
    CIACache().set_cia_path(CIA_path)

    # Planets
    planet = Planet(planet_radius=1, planet_mass=1,)

    # Stars
    star = BlackbodyStar(radius=1,)

    # Chemistry
    chemistry = TaurexChemistry(fill_gases=['H2', 'He'], ratio=he_h2)
    chemistry.addGas(ConstantGas('H2O', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CH4', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CO2', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('CO', mix_ratio=10**-5))
    chemistry.addGas(ConstantGas('NH3', mix_ratio=10**-5))

    # Temperature structure
    IsoT = Isothermal(1000)

    # Putting everything together
    forward_model = TransmissionModel(planet=planet, temperature_profile=IsoT, chemistry=chemistry, star=star,
                                    atm_min_pressure=P_top, atm_max_pressure=P_surface, nlayers=n_layers)

    # enabling contribution from various opacity sources
    forward_model.add_contribution(AbsorptionContribution())
    forward_model.add_contribution(RayleighContribution())
    forward_model.add_contribution(CIAContribution())

    forward_model.build()
    forward_model.model()
    return forward_model

'''
def setup_dedicated_fm(fm, pl_idx, Rs, Mp,ariel_wngrid,ariel_wnwidth ):
    proxy_compute_spectrum = partial(compute_ariel_spectrum, 
                                fm = fm, 
                                Rs = Rs[pl_idx], 
                                Mp = Mp[pl_idx], 
                                ariel_wngrid=ariel_wngrid, 
                                ariel_wnwidth=ariel_wnwidth )
    return proxy_compute_spectrum
'''

# MODIFICA:

def setup_dedicated_fm(fm, X, Rs, Mp, ariel_wngrid, ariel_wnwidth ):
    def proxy_compute_spectrum(params):
        
        # DEBUG: print input to function
        # print("Input parameters:",params) 
        
        for i in range(len(params)):
          if np.isnan(params[i]) or np.isinf(params[i]):
             print(f"Invalid value in input parameters at index {i}:", params)
             return None # Returning None if inputs have nan

        # Example of checking before log
        try:
          radius =  params[0] 
          if radius <= 0:
            print("Invalid radius parameter ", radius)
            return None
        except Exception as e:
           print(e)
           return None

        try:
            spectrum, path_length = fm.compute_spectrum(params, Rs[X], Mp[X], ariel_wngrid, ariel_wnwidth )
        except Exception as e:
            print('Error calculating spectrum', e)
            return None
        #DEBUG
        # print("path_length values:",path_length)
        for i in range(len(path_length)):
            for j in range(len(path_length[i])):
                if np.isnan(path_length[i][j]) or np.isinf(path_length[i][j]):
                    print(f"Invalid value in path_length at layer {i} and index {j}:", path_length[i][j])
                    return None # Returning None if pathlength has nan
        return spectrum, path_length
    return proxy_compute_spectrum


def ariel_resolution():
    import numpy as np
    from taurex.util.util import wnwidth_to_wlwidth
    
    #griglie di lunghezze d'onda
    wlgrid = np.array([0.55      , 0.7       , 0.95      , 1.156375  , 1.27490344,
       1.40558104, 1.5496531 , 1.70849254, 1.88361302, 1.9695975 ,
       2.00918641, 2.04957106, 2.09076743, 2.13279186, 2.17566098,
       2.21939176, 2.26400154, 2.30950797, 2.35592908, 2.40328325,
       2.45158925, 2.50086619, 2.5511336 , 2.60241139, 2.65471985,
       2.70807972, 2.76251213, 2.81803862, 2.8746812 , 2.93246229,
       2.99140478, 3.05153202, 3.11286781, 3.17543645, 3.23926272,
       3.30437191, 3.37078978, 3.43854266, 3.50765736, 3.57816128,
       3.65008232, 3.72344897, 4.03216667, 4.30545796, 4.59727234,
       4.90886524, 5.24157722, 5.59683967, 5.97618103, 6.3812333 ,
       6.81373911, 7.2755592 ])[::-1]
    wlwidth = np.array([0.10083333, 0.20416667, 0.30767045, 0.11301861, 0.12460302,
       0.13737483, 0.15145575, 0.16697996, 0.18409541, 0.03919888,
       0.03998678, 0.04079051, 0.0416104 , 0.04244677, 0.04329995,
       0.04417028, 0.0450581 , 0.04596377, 0.04688764, 0.04783008,
       0.04879147, 0.04977218, 0.0507726 , 0.05179313, 0.05283417,
       0.05389614, 0.05497945, 0.05608453, 0.05721183, 0.05836179,
       0.05953486, 0.06073151, 0.06195222, 0.06319746, 0.06446773,
       0.06576353, 0.06708537, 0.06843379, 0.06980931, 0.07121248,
       0.07264385, 0.07410399, 0.26461764, 0.28255283, 0.30170364,
       0.32215244, 0.34398722, 0.36730191, 0.39219681, 0.41877904,
       0.44716295, 0.47747067])[::-1]
    wngrid = 10000/wlgrid
    wnwidth = wnwidth_to_wlwidth(wlgrid,wlwidth )
    return wlgrid, wlwidth, wngrid, wnwidth

def preprocessing(input, error_flag= False):
    """Preprocess the input before feeding it into the main computation loop (unused)"""
    return input, error_flag

def check_output(q1, q2, q3, min_val = 1e-8, max_val = 0.1):
    """performs two kinds of check, check for nan , inf and -inf, another checks for large finite values """
    q1 = np.nan_to_num(q1, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q2 = np.nan_to_num(q2, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q3 = np.nan_to_num(q3, copy=True, nan=max_val, posinf=max_val, neginf=max_val)
    q1 = np.clip(q1, min_val, max_val)
    q2 = np.clip(q2, min_val, max_val)
    q3 = np.clip(q3, min_val, max_val)

    assert np.sum(~np.isfinite(q1)) == 0, "Unphysical values identified"
    assert np.sum(~np.isfinite(q2)) == 0, "Unphysical values identified"
    assert np.sum(~np.isfinite(q3)) == 0, "Unphysical values identified"
    return q1, q2, q3

'''