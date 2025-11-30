from numpy import exp, abs, log
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class ReedMF(HalosMassFunction):
    """Reed halo mass function (2003, 2007).

    Provides improved fitting at high redshifts by adding a correction
    term to the Jenkins form.
    """

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Reed et al.

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
        fst = 0.315 * exp(-abs(log(1 / sgm) + 0.61) ** 3.8) * (1 + 0.06 / (sgm ** 2.0) ** 0.1)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm