from typing import Annotated
import numpy as np
from pydantic import BaseModel
from starforge.cosmology.cosmologicalmodel import CosmologicalModel

# constants
SEC_PER_YEAR = 3.154e7
M_SUN_GRAM = 1.98847e33
L_SUN_ERG_S = 3.828e33  # Luminosidade solar (IAU 2015)


class ParameterAccretionRate(BaseModel):
    """Container for black hole accretion model parameters.

    Attributes:
        lb_mean_par (float): Characteristic bolometric luminosity in L_sun.
        mbh_par (float): Characteristic black hole mass in solar masses.
        alpha_par (float): Power-law index of the luminosity–mass relation.
        tau_par (float): Characteristic timescale (seconds).
        eta_par (float): Radiative efficiency (dimensionless).
        speed_light (float): Speed of light in cm/s.
    """

    lb_mean_par: Annotated[float,
                           "Characteristic bolometric luminosity (L_sun)"] = 7.96e13
    mbh_par: Annotated[float,
                       "Characteristic black hole mass (M_sun)"] = 2.19e11
    alpha_par: Annotated[float, "Power-law index"] = 2.71e-1
    tau_par: Annotated[float,
                       "Tau (seconds!). Gyr"] = 4.81e9
    eta_par: Annotated[float, "Radiative efficiency (dimensionless)"] = 0.1
    speed_light: Annotated[float, "Speed of light (cm/s)"] = 2.99792458e10
    t_q_par: Annotated[float, "Quasar lifetime (Gyr)"] = 1.0e9
    b1_par: Annotated[float, "Parameter b1"] = 0.0
    b2_par: Annotated[float, "Parameter b2"] = 0.0


class BlackHolesAccretionRate:
    """Black hole accretion model using L_b in solar luminosities (L_sun)."""

    def __init__(self, cosmology: CosmologicalModel, params: ParameterAccretionRate = ParameterAccretionRate()):
        self.params = params
        self.cosmology = cosmology
        self.c_0 = None

    def _broadcast(self, z, mbh):
        z_arr = np.atleast_1d(z)
        mbh_arr = np.atleast_1d(mbh)
        t_z = self.cosmology.age(z_arr)
        return z_arr, mbh_arr, t_z

    def mean_bolometric_luminosity(self, z, mbh):
        """Mean bolometric luminosity L_b(M, z) in erg/s.

        lb_mean_par is given in L_sun and converted to erg/s internally.
        """
        z = np.atleast_1d(z)
        mbh = np.atleast_1d(mbh)
        t_z = self.cosmology.age(z)

        tau = self.params.tau_par
        # convert lb_mean_par from L_sun to erg/s
        lb_mean_cgs = self.params.lb_mean_par * L_SUN_ERG_S

        logL = np.log(lb_mean_cgs) \
            + self.params.alpha_par * (np.log(mbh) - np.log(self.params.mbh_par)) \
            + np.log(tau) - np.log(t_z) - t_z / tau
        L = np.exp(logL)
        return L  # in erg/s

    def f_radiative_efficiency(self, z):
        """
        Compute the redshift-dependent radiative-efficiency function f(z) = eta(z) / (1 - eta(z)).

        Parameters
        ----------
        z : float
            Redshift at which to evaluate the function.

        Returns
        -------
        float
            f(z) = eta(z) / (1 - eta(z)), where eta(z) is the radiative efficiency.
            To obtain the radiative efficiency itself use eta(z) = f(z) / (1 + f(z)).

        Description
        -----------
        This routine implements a phenomenological redshift dependence for the
        ratio f(z) = eta(z) / (1 - eta(z)) of the form

            f(z) = C0 * [ (t_u(z) / t_q)^{b1} + (t_q / t_u(z))^{b2} ]^{-1},

        where
        - t_u(z) is the age of the universe at redshift z (computed via self.cosmology.age(z)),
        - t_q, b1, b2 and eta_par are taken from self.params,
        - C0 is a normalization chosen so that eta(z=0) = eta_par.

        The normalization C0 is therefore

            C0 = (eta_par / (1 - eta_par)) * [ (t_u(0) / t_q)^{b1} + (t_q / t_u(0))^{b2} ].

        Given f(z) = eta(z) / (1 - eta(z)), the radiative efficiency is recovered by

            eta(z) = f(z) / [1 + f(z)].
            f(z) = C0 / [ (t_u(z) / t_q)^{b1} + (t_q / t_u(z))^{b2} ].
            C0 = f(z=0) * [ (t_u(0) / t_q)^{b1} + (t_q / t_u(0))^{b2} ].


        Notes
        -----
        - The function caches C0 on first call (stored as self.c_0) to avoid
          recomputing the normalization.
        - Ensure units of t_q and the cosmology age are consistent.
        """
        def f_z_inverse(t_u):
            term1 = (t_u / self.params.t_q_par) ** self.params.b1_par
            term2 = (self.params.t_q_par / t_u) ** self.params.b2_par
            return (term1 + term2)

        if self.c_0 is None:
            self.c_0 = (self.params.eta_par / (1.0 - self.params.eta_par)
                        ) * f_z_inverse(self.cosmology.age(0.0))

        return self.c_0 / f_z_inverse(self.cosmology.age(z))

    def mean_accretion_rate(self, z, mbh):
        """Mean mass accretion rate in M_sun/yr."""
        L = self.mean_bolometric_luminosity(z, mbh)  # erg/s
        const = (1.0 / (self.params.speed_light ** 2.0)) / \
            self.f_radiative_efficiency(z)
        mdot_cgs = const * L  # g/s
        dot_msun_per_yr = mdot_cgs * SEC_PER_YEAR / M_SUN_GRAM
        return dot_msun_per_yr

    def d_mean_accretion_rate_dmbh(self, z, mbh):
        mdot = (self.params.alpha_par / mbh) * self.mean_accretion_rate(z, mbh)
        return mdot

    def drift_velocity(self, z, mbh, dln_mbh):
        dmd_dm = self.d_mean_accretion_rate_dmbh(z, mbh)
        mdot = self.mean_accretion_rate(z, mbh)
        return dmd_dm + mdot / (mbh * dln_mbh)

    def normalized_accretion(self, z, mbh, dln_mbh):
        norm = self.mean_accretion_rate(z, mbh) / (mbh * dln_mbh)
        return norm
