from numpy import exp
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class WarrenMF(HalosMassFunction):
    """Warren halo mass function (2006).

    Empirical fit from N-body simulations across a wide mass range.
    """

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Warren et al. (2006).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.

        Raises:
            NameError: If halo mass is outside the valid range (10 < log10(M) < 15).
        """
        A = 0.7234
        a = 1.625
        b = 0.2538
        c = 1.1982
        gte = self.perturbations.growth_function(z)
        rdmt = self.cosmology.rodm(z)
        deltac = self.cosmology.params.deltac
        step = lm / 20.0
        kmass = 10.0 ** lm
        if lm < 10 or lm > 15:
            raise NameError("Mass of dark halo outside of the valid range")
        sgm = self.sigma.fstm(lm)
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        sigma1 = deltac / (sgm * gte)
        sigma2 = sigma1 ** 2.0
        fst = A * ((sigma1 ** (-a)) + b) * exp(-c / sigma2)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm