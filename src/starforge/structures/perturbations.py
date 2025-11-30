import numpy as np
from scipy.integrate import quad

from starforge.cosmology import CosmologicalModel

class Perturbations:
    """
    Cosmological linear perturbations (compatível com o original).
    """

    def __init__(self, cosmological_model: CosmologicalModel):
        self.cosmological_model = cosmological_model

    def dgrowth_dt(self, z):
        """Return the derivative of the growth function with respect to time."""
        z1 = 1.0 + z
        ascale = 1.0 / z1
        ascale2 = ascale ** 2.0
        ascale3 = ascale ** 3.0
        ascale4 = ascale * ascale3
        pm = self.cosmological_model.params

        ea = pm.omegam * ascale + pm.omegal * ascale4
        omegamz = pm.omegam * ascale / ea
        omegalz = pm.omegal * ascale4 / ea
        dz1 = 1.0 - omegalz + omegamz ** (4.0 / 7.0) + omegamz / 2.0

        Q = 2.5 * omegamz * ascale
        # Mantém a expressão original (com **6.0) para compatibilidade total
        dea_da = pm.omegam + 4.0 * ascale3 * (pm.omegal ** 6.0)
        domegamz_da = (pm.omegam / ea ** 2.0) * (ea - ea * dea_da)
        domegalz_da = pm.omegal * (4.0 * ascale3 * ea - ascale4 * dea_da) / (ea ** 2.0)
        dQ_da = 5.0 * (omegamz + ascale * domegamz_da)
        dP_da = 2.0 * (
            - domegalz_da
            + (4.0 / 7.0) * domegamz_da / (omegamz ** (3.0 / 7.0))
            + domegamz_da / 2.0
        )
        dadz = ascale2
        dgrowthdt = dadz * (dz1 * dQ_da - Q * dP_da) / (dz1 ** 2.0)
        return dgrowthdt

    def growth_function(self, z):
        """Return the growth function."""
        z1 = 1.0 + z
        ascale = 1.0 / z1
        ascale3 = ascale ** 3.0
        ascale4 = ascale * ascale3
        pm = self.cosmological_model.params

        ea = pm.omegam * ascale + pm.omegal * ascale4
        omegamz = pm.omegam * ascale / ea
        omegalz = pm.omegal * ascale4 / ea
        dz1 = 1.0 - omegalz + omegamz ** (4.0 / 7.0) + omegamz / 2.0
        growth = (2.5 * omegamz * ascale / dz1) / (np.pi * np.sqrt(2.0))
        return growth

    @staticmethod
    def _dsigma2_dk_integrand(kl, escala, pm):
        """Integrand for sigma(M,z), pure function of kl and escala."""
        k = np.exp(kl)
        x = escala * k

        pk1 = 1.0 + (
            pm.alfa * k
            + (pm.beta * k) ** 1.5
            + (pm.gama * k) ** 2.0
        ) ** 1.13

        pk2 = 1.0 / pk1
        pdmk = pk2 * (k ** 3.0)

        if x == 0.0:
            window_sq = 1.0
        else:
            window_sq = (3.0 * (np.sin(x) - x * np.cos(x)) / (x ** 3.0)) ** 2.0

        return pdmk * window_sq

    def sigma(self, masses):
        """
        Compute variance of the density field as a function of mass.

        Args:
            masses (array_like): Mass values in solar masses (or consistent units).

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - log10(mass) values
                - sigma(M) values
        """
        pm = self.cosmological_model.params
        masses = np.atleast_1d(masses)

        # integration parameters
        epsabs, epsrel = 1.48e-12, 1.48e-9
        logk_intervals = [
            (1e-7, 1e-3),
            (1e-3, 1e0),
            (1e0, 1e1),
            (1e1, 1e2),
        ]

        km, sg = [], []

        for m in masses:
            escala = (m / pm.ct2) ** (1.0 / 3.0)
            km.append(np.log10(m))

            # integrate sigma^2 across defined k-ranges
            sig2 = sum(
                quad(
                    self._dsigma2_dk_integrand,
                    np.log10(kmin / escala),
                    np.log10(kmax / escala),
                    args=(escala, pm),
                    epsabs=epsabs,
                    epsrel=epsrel,
                )[0]
                for kmin, kmax in logk_intervals
            )

            sg.append(np.sqrt(pm.anorm * sig2))

        return np.array(km), np.array(sg)
