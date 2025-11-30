from numpy import exp
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class WatsonWT1MF(HalosMassFunction):
    """Watson WT1 halo mass function (2013).

    Provides an empirical fit for halos defined with a spherical overdensity
    criterion. Valid in the redshift range z = 0–30.
    """

    def __init__(self, cosmology, perturbations, sigma, validadeMassRange):
        """
        Args:
            cosmology: Cosmology object.
            sigma: Variance provider object σ(M).
            validadeMassRange (Callable): Function to check valid mass range.
        """
        super().__init__(cosmology, perturbations, sigma)
        self.validadeMassRange = validadeMassRange

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Watson WT1 (2013).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.
        """
        A, a, b, c = 0.282, 2.163, 1.406, 1.21
        gte = self.perturbations.growth_function(z)
        rdmt = self.cosmology.rodm(z)
        step = lm / 20.0
        kmass = 10.0 ** lm
        sgm = self.sigma.fstm(lm)
        self.validadeMassRange(sgm, -0.55, 1.31)
        sgmD = sgm * gte
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        fst = A * (((b / sgmD) ** a) + 1.0) * exp(-c / sgmD ** 2.0)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm