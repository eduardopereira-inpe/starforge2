from numpy import exp
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class WatsonWT2MF(HalosMassFunction):
    """Watson WT2 halo mass function (2013).

    Alternative parameterization of the Watson fit,
    with explicit redshift dependence and cosmological corrections.
    """

    def __init__(self, cosmology, perturbations, sigma, deltaWT, validadeMassRange):
        """
        Args:
            cosmology: Cosmology object.
            sigma: Variance provider object σ(M).
            deltaWT (float): Overdensity definition for halos.
            validadeMassRange (Callable): Function to check valid mass range.
        """
        super().__init__(cosmology, perturbations, sigma)
        self.deltaWT = deltaWT
        self.validadeMassRange = validadeMassRange

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Watson WT2 (2013).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.

        Raises:
            NameError: If redshift is negative.
        """
        gte = self.perturbations.growth_function(z)
        rdmt = self.cosmology.rodm(z)
        step = lm / 20.0
        kmass = 10.0 ** lm
        sgm = self.sigma.fstm(lm)
        sgmD = sgm * gte
        omz = self.cosmology.omegamz(z)
        if z < 0:
            raise NameError("z lower than zero.")
        if z == 0:
            self.validadeMassRange(sgm, -0.55, 1.05)
            A, a, b, gm = 0.194, 2.267, 1.805, 1.287
        elif z >= 6:
            A, a, b, gm = 0.563, 3.810, 0.874, 1.453
        else:
            A = omz * (1.097 * (1.0 + z) ** (-3.216) + 0.074)
            a = omz * (5.907 * ((1.0 + z) ** (-3.058)) + 0.46)
            b = omz * (3.136 * ((1.0 + z) ** (-1.970)) + 0.42)
            gm = omz * (1.438 * ((1.0 + z) ** (-0.505)) + 0.046)
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        fst = A * ((sgmD / b) ** (-a) + 1) * exp(-gm / sgmD ** 2)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm