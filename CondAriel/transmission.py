"""Transit forward model."""
import csv
import typing as t
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import os

from taurex.chemistry import Chemistry
from taurex.planet import Planet
from taurex.pressure import PressureProfile
from taurex.stellar import Star
from taurex.temperature import TemperatureProfile

from .simplemodel import OneDForwardModel

if t.TYPE_CHECKING:
    from taurex.contributions import Contribution
else:
    Contribution = object


class TransmissionModel(OneDForwardModel):
    """A forward model for transits."""
    def __init__(
        self,
        planet: t.Optional[Planet] = None,
        star: t.Optional[Star] = None,
        pressure_profile: t.Optional[PressureProfile] = None,
        temperature_profile: t.Optional[TemperatureProfile] = None,
        chemistry: t.Optional[Chemistry] = None,
        nlayers: t.Optional[int] = 100,
        atm_min_pressure: t.Optional[float] = 1e-4,
        atm_max_pressure: t.Optional[float] = 1e6,
        contributions: t.Optional[t.List[Contribution]] = None,
        new_path_method: t.Optional[bool] = False,
    ) -> None:
        """Initialize transit forward model.

        Parameters
        ----------
        name: str
            Name to use in logging

        planet:
            Planet model, default planet is Jupiter

        star:
            Star model, default star is Sun-like

        pressure_profile:
            Pressure model, alternative is to set ``nlayers``, ``atm_min_pressure``
            and ``atm_max_pressure``

        temperature_profile:
            Temperature model, default is an
            :class:`~taurex.data.profiles.temperature.isothermal.Isothermal`
            profile at 1500 K

        chemistry:
            Chemistry model, default is
            :class:`~taurex.data.profiles.chemistry.taurexchemistry.TaurexChemistry`
            with ``H2O`` and ``CH4``

        nlayers: int, optional
            Number of layers. Used if ``pressure_profile`` is not defined.

        atm_min_pressure: float, optional
            Pressure at TOA. Used if ``pressure_profile`` is not defined.

        atm_max_pressure: float, optional
            Pressure at BOA. Used if ``pressure_profile`` is not defined.

        contributions: list, optional
            List of contributions to include

        new_path_method: bool, optional
            Use new path length computation method

        """
        super().__init__(
            self.__class__.__name__,
            planet,
            star,
            pressure_profile,
            temperature_profile,
            chemistry,
            nlayers,
            atm_min_pressure,
            atm_max_pressure,
            contributions,
        )
        self.new_method = new_path_method

    def layers(self) -> int:
        """Returns the number of layers in the model."""
        return self.nLayers
    # NEW VERSION
    def compute_path_length_old(self, dz: npt.NDArray[np.float64]) -> t.List[npt.NDArray[np.float64]]:
        """Compute path length for each layer."""
        # print("Compute_path_length_old invocato")
        dl = []
        planet_radius = self._planet.fullRadius
        total_layers = self.nLayers
        z = self.altitudeProfile

        # print(f"planet_radius: {planet_radius}m total_layers: {total_layers}")
        # print(F"dz: {dz}")
        # print(f"z: {z}")
        
        # Open CSV file for writing k values
        with open('path_length_k_values.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Layer', 'k[0]', 'k[1:]'])  # Header row
            for layer in range(0, total_layers):
                sqrt_arg = planet_radius + dz[0] / 2 + z[layer:self.nLayers-1] + dz[layer:self.nLayers-1] / 2
                if np.any(np.isnan(sqrt_arg)) or np.any(sqrt_arg < 0):
                    # print(f"Errore all'iterazione: {layer}, argomento radice < 0 o Not-a-Number")
                    continue
                # if np.any(np.abs(planet_radius + dz[0] / 2 + z[layer]) > 1e5): 
                    # print(f"Large value detected at layer {layer}: {planet_radius + dz[0] / 2 + z[layer]}")  
                p = (planet_radius + dz[0] / 2 + z[layer]) ** 2
                try:
                    k = np.zeros(shape=(self.nLayers - layer))
                    k[0] = np.sqrt((planet_radius + dz[0] / 2.0 + z[layer] + dz[layer] / 2.0) ** 2 - p)
                    if k[0] < 0 or np.isnan(k[0]) or np.isinf(k[0]):
                        raise ValueError(f"Invalid value for k[0]: {k[0]}")
                    k[1:] = np.sqrt((planet_radius + dz[0] / 2 + z[layer + 1:] + dz[layer + 1:] / 2) ** 2 - p)
                    k[1:] -= np.sqrt((planet_radius + dz[0] / 2 + z[layer:self.nLayers - 1] + dz[layer:self.nLayers - 1] / 2) ** 2 - p)
                    '''
                    # Plot k[0] separately
                    plt.figure(figsize=(10,6))
                    plt.plot([0], [k[0]], 'ro', label='k[0]')
                    plt.plot(range(1, len(k)), k[1:], 'b-', label='k[1:]')
                    plt.xlabel('Index')
                    plt.ylabel('planet_k_vals')
                    plt.title(f'Planet k values for Layer {layer}')
                    plt.legend()
                    plt.grid(True)
                    filename = f'./k_vals_plots/path_length_layer_{layer}_detailed.png'
                    plt.savefig(filename)
                    plt.close()
                    plt.figure(figsize=(10,6))
                    plt.plot(k,label=f'Layer {layer}')
                    plt.xlabel('Index')
                    plt.ylabel('path_length k vals')
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(f'./path_length_images/path_length_layer_{layer}.png')
                    # plt.show()
                    plt.close()
                    '''
                    # Evitiamo temporaneamente problemi sui dati, sostituendo quelli problematici (<0,NaN,INF) con zeri
                    k[1:][k[1:] < 0] = 0
                    k[1:][np.isnan(k[1:])] = 0
                    k[1:][np.isinf(k[1:])] = 0
                    # Write k values to CSV
                    writer.writerow([layer, k[0], k[1:].tolist()])
                    if np.any(np.isnan(k)):
                        # print(f"Not-a-Number found")
                        continue
                    dl.append(k * 2.0)
                except ValueError as e:
                    # print(f"Errore durante il calcolo per layer: {layer}: {e}")
                    continue 
        return dl
    
    def compute_path_length(self) -> t.List[npt.NDArray[np.float32]]:
        """Compute path length for each layer, new method."""
        from taurex.util.geometry import parallel_vector
        altitude_boundaries = self.altitude_boundaries
        radius = self.planet.fullRadius
        # Generate our line of sight paths
        viewer, tangent = parallel_vector(radius, self.altitude_profile + self.deltaz / 2, altitude_boundaries.max())
        path_lengths = self.planet.compute_path_length(altitude_boundaries, viewer, tangent)
        return [l for _, l in path_lengths]

    def path_integral(self, wngrid: npt.NDArray[np.float32], return_contrib: t.Optional[bool] = False) -> t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Compute path integral.
        Calculates the absorption and optical depth for each layer assuming
        hemispherical geometry.
        """
        dz = self.deltaz
        total_layers = self.nLayers
        wngrid_size = wngrid.shape[0]
        density_profile = self.densityProfile
        if self.new_method:
            path_length = self.compute_path_length()
        else:
            path_length = self.compute_path_length_old(dz)
        self.path_length = path_length
        tau = np.zeros(shape=(total_layers, wngrid_size), dtype=np.float32)
        # print(f"Layers totali: {total_layers}, path_length: {len(path_length)}")
        # print(f"path_length values: {path_length}")
        for layer in range(total_layers):
            # self.debug("Computing layer %s", layer)
            if(layer>=len(path_length)):
                print(f"Skipping layer {layer} due to insufficient path length")
                continue
            dl = path_length[layer]
            # DEBUG: print(f"Layer: {layer},  path_length val: {dl}")
            end_k = total_layers - layer
            for contrib in self.contribution_list:
                if tau[layer].min() > 10:
                    break
                self.debug("Adding contribution from %s", contrib.name)
                contrib.contribute(
                    self, 0, end_k, layer, layer, density_profile, tau, path_length=dl
                )
        self.debug("tau %s %s", tau, tau.shape)
        absorption, tau = self.compute_absorption(tau, dz)
        return absorption, tau

    def compute_absorption(
        self, tau: npt.NDArray[np.float32], dz: npt.NDArray[np.float32]
    ) -> t.Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Compute final absorption and optical depth."""
        tau = np.exp(-tau)
        ap = self.altitudeProfile[:, None]
        pradius = self._planet.fullRadius
        sradius = self._star.radius
        _dz = dz[:, None]

        integral = np.sum((pradius + ap) * (1.0 - tau) * _dz * 2.0, axis=0)
        return ((pradius**2.0) + integral) / (sradius**2), tau

    @classmethod
    def input_keywords(cls) -> t.Tuple[str, ...]:
        """Input keywords for this class."""
        return (
            "transmission",
            "transit",
        )