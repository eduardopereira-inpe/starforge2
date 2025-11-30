from numpy import exp, abs, log
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class JenkinsMF(HalosMassFunction):
    """Jenkins halo mass function (2001).

    Empirical fit to N-body simulations.
    Depends only on σ(M).
    """

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Jenkins et al. (2001).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.
        """
        sgm = self.sigma.fstm(lm)
        rdmt = self.cosmology.rodm(z)
        step = lm / 20.0
        kmass = 10.0 ** lm
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        fst = 0.315 * exp(-abs(log(1.0 / sgm) + 0.61) ** 3.8)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm