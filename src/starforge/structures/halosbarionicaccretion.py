from scipy.integrate import quad
import scipy.interpolate as spint
from scipy.interpolate import InterpolatedUnivariateSpline

import numpy as np

from starforge.utils import locate
from starforge.cosmology.cosmologicalmodel import CosmologicalModel
from .massfunctions import HalosMassFunction


class HaloBaryonAccretion:
    """
    Compute dark matter halo and baryonic matter properties related to
    structure formation.

    This class provides tools to calculate:
    - Dark halo mass function integrals.
    - Numerical density of halos.
    - Fraction of baryons bound to structures.
    - Baryonic accretion rate into halos as a function of scale factor.

    Attributes
    ----------
    massfunction : HalosMassFunction
        The halo mass function model used for calculations.
    _cosmology : CosmologicalModel
        The cosmological model attached to the mass function.
    _zmax : float
        Maximum redshift for computations.
    _abt2 : np.ndarray or None
        Baryonic accretion rate values on the computed scale-factor grid.
    _ascale : np.ndarray or None
        Scale-factor grid used for interpolation.
    _tck_ab : tuple or None
        Spline representation of the baryonic accretion rate.
    __lmInf, __lmSup : float
        Lower and upper bounds of log10 halo mass (from sigma object).
    """

    def __init__(self, massfunction: HalosMassFunction, zmax: float = 10.0):
        self.massfunction = massfunction
        self._cosmology: CosmologicalModel = massfunction.cosmology
        self._zmax = zmax

        self._abt2 = None
        self._ascale = None
        self._tck_ab = None

        # mass limits (log10(M))
        self.__lmInf, self.__lmSup = (
            self.massfunction.sigma.lmInf,
            self.massfunction.sigma.lmSup,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def abt2(self):
        """Baryonic accretion rate array on the scale-factor grid."""
        return self._abt2

    @property
    def ascale(self):
        """Scale-factor grid used for interpolation."""
        return self._ascale

    @property
    def tck_ab(self):
        """Spline representation of baryonic accretion rate vs scale factor."""
        return self._tck_ab

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------
    def mass_function_M(self, lnM: float, z: float) -> float:
        """
        Mass function integrand in terms of log10(M).

        Parameters
        ----------
        lnM : float
            log10(M / Msun), halo mass in base-10 logarithm.
        z : float
            Redshift.

        Returns
        -------
        float
            Value of the integrand f(σ) * M * ln(10).
        """
        M = 10.0**lnM
        f_sigma = self.massfunction.massFunction(lnM, z)
        tilt2 = self._cosmology.params.tilt / np.log(10.0)
        return tilt2 * f_sigma * M**2
    

    def halos_n(self, z: float) -> float:
        """
        Compute the integrated halo mass density at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Integrated mass density of halos.
        """
        fmassM = lambda lm: self.mass_function_M(lm, z)
        result, _ = quad(fmassM, self.__lmInf, self.__lmSup, epsrel=1e-6)
        return result 
  

    def fbstruc(self, z: float) -> float:
        """
        Fraction of baryons bound to collapsed halos at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Fraction of baryons in structures.
        """
        return self.halos_n(z) / self._cosmology.rodm(z)

    def numerical_density_halos(self, z: float) -> float:
        """
        Compute the comoving number density of halos at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        float
            Number density of halos.
        """
        f = lambda lm: self.massfunction.massFunction(lm, z)
        result, _ = quad(f, self.__lmInf, self.__lmSup, epsrel=1e-5)
        return result

    def abt(self, a: float) -> float:
        """
        Interpolate the baryonic accretion rate at scale factor a.

        Parameters
        ----------
        a : float
            Scale factor (a = 1 / (1 + z)).

        Returns
        -------
        float
            Baryonic accretion rate at scale factor a.
        """
        i = locate(self._ascale, len(self._ascale) - 1, a)
        return self._abt2[i]

    def start_baryonic_accretion_rate(self, np_steps=1000, method="spline"):
        """
        Compute baryonic accretion rate:
            ρ̇_b(z) = Ω_b ρ_c,0 |df_b/dz| (dt/dz)^{-1}

        Args:
            np_steps (int): Number of points in z-grid.
            method (str): Derivative method: "spline" or "gradient".
        """
        # redshift grid (crescente para spline)
        z_grid = np.linspace(0, self._zmax, np_steps + 1)
        ascale = 1.0 / (1.0 + z_grid)

        # baryonic fraction
        fbt2 = np.array([self.fbstruc(zi) for zi in z_grid])

        # derivative
        if method == "spline":
            spline = InterpolatedUnivariateSpline(z_grid, fbt2, k=3)
            dfdz = spline.derivative()(z_grid)
        elif method == "gradient":
            dfdz = np.gradient(fbt2, z_grid)
        else:
            raise ValueError("method must be 'spline' or 'gradient'")

        # accretion rate
        dt_dz = np.array([self._cosmology.dt_dz(zi) for zi in z_grid])
        abt2 = self._cosmology.params.robr0 * np.abs(dfdz) / dt_dz

        # save results (invert so ascale goes 1 → 0, like before)
        self._ascale = ascale[::-1]
        self._abt2 = abt2[::-1]
        self._tck_ab = InterpolatedUnivariateSpline(self._ascale, self._abt2, k=3)

