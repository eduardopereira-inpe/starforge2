from numpy import exp
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class BurrMF(HalosMassFunction):
    """Burr halo mass function.

    Uses a Burr-type distribution as an alternative parameterization
    for the halo mass function.
    """

    def __init__(self, cosmology, perturbations, sigma, qBurr, burrBq):
        """
        Args:
            cosmology: Cosmology object.
            sigma: Variance provider object σ(M).
            qBurr (float): Burr distribution shape parameter q.
            burrBq (Callable): Function returning normalization factor B(q).
        """
        super().__init__(cosmology, perturbations, sigma)
        self.qBurr = qBurr
        self.burrBq = burrBq

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM using the Burr mass function.

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.
        """
        rdmt = self.cosmology.rodm(z)
        step = lm / 20.0
        kmass = 10.0 ** lm
        sgm = self.sigma.fstm(lm)
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        Bq = self.burrBq(self.qBurr)
        fst = (1 + sgm) ** (-1 - 1 / (1 - self.qBurr)) / Bq
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm