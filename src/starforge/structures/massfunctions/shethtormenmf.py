from numpy import exp, sqrt, pi
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class ShethTormenMF(HalosMassFunction):
    """Sheth & Tormen halo mass function (1999).

    Extends the Press-Schechter formalism by accounting for ellipsoidal collapse,
    producing better agreement with N-body simulations.

    Attributes:
        ctst (float): Normalization constant.
        ast2 (float): Shape parameter controlling the exponential cutoff.
        pst (float): Empirical slope correction.
    """

    def __init__(self, cosmology, perturbations, sigma):
        """
        Args:
            cosmology: Cosmology object.
            sigma: Variance provider object σ(M).
            ctst (float): Normalization factor.
            ast2 (float): Exponential cutoff parameter.
            pst (float): Power-law slope parameter.
        """
        super().__init__(cosmology, perturbations, sigma)

        self.ast2 = 0.707
        self.pst = 0.3
        self.ast1 = 0.322
        self.ctst = self.ast1 * sqrt(2.0 * self.ast2 / pi)
        print("Using ST Mass Function")

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Sheth & Tormen (1999).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.
        """
        gte = self.perturbations.growth_function(z)
        rdmt = self.cosmology.rodm(z)
        deltac = self.cosmology.params.deltac
        step = lm / 20.0
        kmass = 10.0 ** lm
        sgm = self.sigma.fstm(lm)
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        sigma1 = deltac / (sgm * gte)
        sigma2 = sigma1 ** 2.0
        expn = exp(-self.ast2 * sigma2 / 2.0)
        fst = self.ctst * sigma1 * (1.0 + (1.0 / (sigma2 * self.ast2)) ** self.pst) * expn
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm
