import numpy as np
from numpy import log10, sqrt, array, zeros
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from starforge.utils.run_kut4 import rk4_int
from starforge.structures import HaloBaryonAccretion


class SFR:
    """Cosmic Star Formation Rate (CSFR) model.

    This class implements the cosmic star formation rate following
    Pereira & Miranda (2010), making use of existing cosmological background
    models, halo baryon accretion, and IMF prescriptions.

    It integrates the evolution of gas density inside structures to compute
    the CSFR, using RK4 for numerical integration and modern interpolation
    routines from SciPy.

    Attributes:
        astar (np.ndarray): Scale factor grid used in integration.
        csfr (np.ndarray): Computed cosmic star formation rate values.
        rho_gas (np.ndarray): Gas density in structures across redshift.
    """

    def __init__(
        self,
        halo_acc: HaloBaryonAccretion,
        tau: float = 2.29,  # Gyr
        eimf: float = 1.35,
        nsch: float = 1.0,
        imf_type: str = "S",
        zmax: float = 20.0,
    ) -> None:
        """Initialize the SFR model.

        Args:
            halo_acc (HaloBaryonAccretion): Precomputed halo baryon accretion model.
            tau (float, optional): Star formation timescale in Gyr. Defaults to 2.29.
            eimf (float, optional): IMF slope parameter. Defaults to 1.35.
            nsch (float, optional): Schmidt index (gas density exponent). Defaults to 1.0.
            imf_type (str, optional): IMF type, "S" for Salpeter or "K" for Kroupa.
                Defaults to "S".
            zmax (float, optional): Maximum redshift for integration. Defaults to 20.0.
        """
        self._cosmology = halo_acc.massfunction.cosmology
        self._halo_acc = halo_acc

        # IMF parameters
        self.__nsch = nsch
        self.__eimf = eimf
        self.imf_type = imf_type
        self.__eimf0 = eimf - 1.0

        # Timescale converted to years
        self.__tau = tau * 1.0e9
        self._zmax = zmax

        # Mass integration ranges
        self.__aminf1 = 2.5e1
        self.__amsup1 = 1.4e2
        self.__amin = 1.0e-1

        # IMF selection map
        self.__imf_map = {"S": self.__imf_salpeter, "K": self.__imf_kroupa}

        # Normalization constant for IMF integrals
        self.__anorm1 = None

        # Compute halo baryonic accretion arrays
        self._halo_acc.start_baryonic_accretion_rate()
        self._ascale = self._halo_acc.ascale
        self._abt2 = self._halo_acc.abt2

        # Use modern spline for accretion rate interpolation
        spline_ab = getattr(self._halo_acc, "tck_ab", None)
        if spline_ab is None:
            raise RuntimeError("HaloBaryonAccretion must provide tck_ab (spline).")
        self._spline_ab = spline_ab

        # ESNOR normalization logic (simplified: always normalize globally)
        self._global_normalization = True
        self.__esnor = 1.0

        # Compute the CSFR arrays
        self.__csfr, self.__rho_gas, self.__astar = self.__sfr()

        # Public arrays
        self.astar = self.__astar
        self.csfr = self.__csfr
        self.rho_gas = self.__rho_gas

    # ------------------------------------------------------------------
    # IMF definitions
    # ------------------------------------------------------------------
    def __imf_kroupa(self, m: float) -> float:
        """Kroupa IMF.

        Args:
            m (float): Stellar mass in solar units.

        Returns:
            float: IMF value at given mass.
        """
        if self.__anorm1 is None:
            alpha0, alpha1, alpha2 = 0.3, 1.3, 2.3
            k0 = 1
            k1 = k0 * 0.08
            k2 = k1 * 0.5
            k3 = k2
            A = [
                k0 * (self.__amsup1 ** (1 - alpha0) - self.__amin ** (1 - alpha0)) / (1 - alpha0),
                k1 * (self.__amsup1 ** (1 - alpha1) - self.__amin ** (1 - alpha1)) / (1 - alpha1),
                k2 * (self.__amsup1 ** (1 - alpha2) - self.__amin ** (1 - alpha2)) / (1 - alpha2),
                k3 * (self.__amsup1 ** (1 - alpha2) - self.__amin ** (1 - alpha2)) / (1 - alpha2),
            ]
            self.__anorm1 = 1 / sum(A)

        if 0.08 < m <= 0.5:
            return self.__anorm1 * m**-1.3
        elif m > 0.5:
            return self.__anorm1 * m**-2.3
        raise ValueError("Mass out of range in IMF Kroupa")

    def __imf_salpeter(self, m: float) -> float:
        """Salpeter IMF."""
        if self.__anorm1 is None:
            self.__anorm1 = self.__eimf0 / (
                1.0 / self.__amin**self.__eimf0 - 1.0 / self.__amsup1**self.__eimf0
            )
        return self.__anorm1 * m ** (-(1.0 + self.__eimf))

    def phi(self, m: float) -> float:
        """Evaluate selected IMF at mass `m`."""
        if self.imf_type not in self.__imf_map:
            raise ValueError("Invalid IMF type")
        return self.__imf_map[self.imf_type](m)

    # ------------------------------------------------------------------
    # Stellar remnants and ejected mass
    # ------------------------------------------------------------------
    def remnant(self, m: float) -> float:
        """Remnant mass relation based on stellar mass."""
        if m < 1:
            return 0
        if 1 <= m <= 8:
            return 0.1156 * m + 0.4551
        if 8 < m <= 10:
            return 1.35
        if 10 < m < 25:
            return 1.4
        if 25 <= m <= 145:
            return (13.0 / 24.0) * (m - 20)
        raise ValueError("Mass out of remnant range")

    # ------------------------------------------------------------------
    # Differential equation
    # ------------------------------------------------------------------
    def __fcn(self, a: float, rho_g: np.ndarray) -> np.ndarray:
        """Differential equation for gas density evolution."""
        z = max(0.0, 1.0 / a - 1.0)
        tage = self._cosmology.age(z)

        age01 = 4.0 * log10(tage) - 2.704e1
        age02 = (3.6 - sqrt(age01)) / 2.0

        mi_1 = 10.0**age02
        yr = self.__mass_ejected(mi_1)

        if self.__nsch == 1.0:
            sexp = (1.0 - yr) / self.__tau
        else:
            sexp = (1.0 - yr) / self.__tau / (self._cosmology.params.robr0 ** (self.__nsch - 1.0))

        F = zeros(1)
        F[0] = (
            -sexp * (rho_g[0] ** self.__nsch) + self.__esnor * self._spline_ab(a)
        ) * self._cosmology.dt_dz(z) / a**2
        return F

    # ------------------------------------------------------------------
    # Main integration routine
    # ------------------------------------------------------------------
    def __sfr(self):
        """Integrate the SFR model and return arrays."""
        rho_g0 = array([1.0e-9])
        a0 = self._ascale[0]
        af = self._ascale[-1]
        step = (af - a0) / 100.0

        A, R_g = rk4_int(self.__fcn, a0, rho_g0, af, step)

        # Global normalization (default)
        if self._global_normalization:
            rho_g0 = array([1.0e-9])
            a0 = 1.0 / (self._zmax + 1.0)
            step = (af - a0) / 5000.0
            A, R_g = rk4_int(self.__fcn, a0, rho_g0, af, step)

        rho_s = self.__csfr_gas(R_g)
        return rho_s, R_g, A

    def __csfr_gas(self, rg: np.ndarray) -> np.ndarray:
        """Compute CSFR from gas density."""
        if self.__nsch == 1:
            return rg / self.__tau
        return (rg**self.__nsch) / self.__tau / (
            self._cosmology.params.robr0 ** (self.__nsch - 1.0)
        )
    
        
    # ------------------------------------------------------------------
    # Mass ejection
    # ------------------------------------------------------------------
    def __m_phi(self, m: float) -> float:
        """Return m * phi(m)."""
        return m * self.phi(m)

    def __mr_phi(self, m: float) -> float:
        """Return remnant(m) * phi(m)."""
        return self.remnant(m) * self.phi(m)

    def __mass_ejected_salpeter(self, m_min: float) -> float:
        """Mass ejected for Salpeter IMF (analytic form)."""
        if self.__anorm1 is None:
            self.__imf_salpeter(10)  # ensures normalization is set

        amexp1 = (1.0 / m_min) ** self.__eimf0
        amexp2 = (1.0 / self.__amsup1) ** self.__eimf0
        amexp3 = (1.0 / 8.0) ** self.__eimf0
        amexp4 = (1.0 / self.__aminf1) ** self.__eimf0
        amexp5 = (1.0 / m_min) ** self.__eimf
        amexp6 = (1.0 / 8.0) ** self.__eimf
        amexp7 = (1.0 / 10.0) ** self.__eimf
        amexp8 = (1.0 / self.__aminf1) ** self.__eimf
        amexp9 = (1.0 / self.__amsup1) ** self.__eimf

        yrem1 = (amexp1 - amexp2) / self.__eimf0
        yrem2 = 1.156e-01 * (amexp1 - amexp3) / self.__eimf0
        yrem3 = 1.3e+01 * (amexp4 - amexp2) / self.__eimf0 / 2.4e+01
        yrem4 = 4.551e-01 * (amexp5 - amexp6) / self.__eimf
        yrem5 = 1.35e+00 * (amexp6 - amexp7) / self.__eimf
        yrem6 = 1.40e+00 * (amexp7 - amexp8) / self.__eimf
        yrem7 = 6.5e+01 * (amexp8 - amexp9) / self.__eimf / 6.0

        return self.__anorm1 * (yrem1 - yrem2 - yrem3 - yrem4 - yrem5 - yrem6 + yrem7)

    def __mass_ejected(self, m_min: float) -> float:
        """Return ejected mass fraction for given minimum mass."""
        if self.imf_type == "S":
            return self.__mass_ejected_salpeter(m_min)
        else:
            integral_phi, _ = quad(self.__m_phi, m_min, self.__amsup1, epsrel=1e-5)
            integral_rphi, _ = quad(self.__mr_phi, m_min, self.__amsup1, epsrel=1e-5)
            return (integral_phi - integral_rphi)


    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------
    def cosmic_star_formation_rate(self, z: float) -> float:
        """Get CSFR value at redshift `z`."""
        a = 1.0 / (1.0 + z)
        cspline = CubicSpline(self.__astar, self.__csfr, extrapolate=True)
        return float(cspline(a))

    def gas_density_in_structures(self, z: float) -> float:
        """Get gas density in structures at redshift `z`."""
        a = 1.0 / (1.0 + z)
        rspline = CubicSpline(self.__astar, self.__rho_gas, extrapolate=True)
        return float(rspline(a))
