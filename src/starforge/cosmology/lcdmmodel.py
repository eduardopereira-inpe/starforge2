#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from scipy.integrate import quad
import numpy as np

from .cosmologicalmodel import CosmologicalModel
from .constants import C_LIGHT, AGE_FACTOR
from .cosmologicalparameters import CosmologicalParameters


class LCDMModel(CosmologicalModel):
    """
    LCDM cosmology (Cold Dark Matter + Cosmological Constant).

    Implements the standard LambdaCDM cosmology using parameters provided
    by a Pydantic schema. Provides Hubble parameter, cosmic age, comoving
    distances, comoving volume, and dark/baryonic densities.

    Args:
        params (CosmologyParameters): Pydantic schema with cosmological parameters.
    """

    def __init__(self, params: CosmologicalParameters):
        self.params = params
        self._name = "LCDM"
        self._omegal_m_ratio = self.params.omegal / self.params.omegam

    def H(self, z):
        """
        Hubble parameter at redshift z.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: H(z) in km/s/Mpc.
        """
        return self.params.H0 * np.sqrt(self.params.omegam * (1 + z)**3 + self.params.omegal)

    def dt_dz(self, z):
        """
        Differential cosmic time with respect to redshift.

        Args:
            z (float): Redshift.

        Returns:
            float: dt/dz in years.
        """
        dtdz = self.params.age_factor / (self.params.h *
                                         ((1 + z) *
                                          np.sqrt(
                                              self.params.omegal
                                              + self.params.omegam * (1 + z)**3
                                         )))
        return dtdz

    def dr_dz(self, z):
        """
        Differential comoving distance per unit redshift.

        Args:
            z (float): Redshift.

        Returns:
            float: dr/dz in Mpc.
        """
        # Speed of Light km / s
        vl = 3.0e+5

        # H_{0}/h = 1/s
        hub = 3.25e-18

        drdz = (vl / hub / self.params.h) / np.sqrt(self.params.omegam * (1.0 + z) ** 3.0
                                                    + self.params.omegal)
        return drdz

    def dV_dz(self, z):
        """
        Differential comoving volume element at redshift z.

        Args:
            z (float): Redshift.

        Returns:
            float: dV/dz in Mpc^3 per steradian.
        """
        r, _ = quad(self.dr_dz, 0.0, z)  # integrate dr/dz from 0 to z
        drdz = self.dr_dz(z)
        return 4.0 * np.pi * drdz * r**2

    def rodm(self, z):
        """
        Dark matter density at redshift z.

        Args:
            z (float): Redshift.

        Returns:
            float: Dark matter density in units of rho_crit0.
        """
        a = 1.0 / (1.0 + z)
        return self.params.rodm0 / a**3

    def robr(self, z):
        """
        Baryonic matter density at redshift z.

        Args:
            z (float): Redshift.

        Returns:
            float: Baryonic density in units of rho_crit0.
        """
        a = 1.0 / (1 + z)
        return self.params.robr0 / a**3

    def age(self, z):
        """
        Age of the universe at redshift z.

        Args:
            z (float): Redshift.

        Returns:
            float: Age in years.
        """
        a = 1.0 / (1 + z)
        fct = self._omegal_m_ratio * a**3
        return 6.522916e9 * np.log(np.sqrt(fct) + np.sqrt(fct + 1)) / (self.params.h * np.sqrt(self.params.omegal))

    def omegamz(self, z):
        """
        Matter density parameter Omega_m(z) at redshift z.

        Args:
            z (float): Redshift.

        Returns:
            float: Omega_m(z)
        """
        a = (1.0 / (1.0 + z))
        return self.params.omegam * a / (self.H(z)/self.params.H0)**2

    def luminosity_distance(self, z):
        """
        Compute the luminosity distance at redshift z.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Luminosity distance in parsecs.
        """
        r, _ = quad(self.dr_dz, 0.0, z)  # integrate dr/dz from 0 to z
        d_L = (1 + z) * r  # luminosity distance in Mpc
        return d_L

    def comoved_volume(self, z):
        """
        Compute the comoving volume out to redshift z.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Comoving volume in Mpc^3.
        """
        V, _ = quad(self.dV_dz, 0.0, z)  # integrate dV/dz from 0 to z
        return V
