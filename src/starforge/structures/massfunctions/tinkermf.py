from numpy import exp, log
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class TinkerMF(HalosMassFunction):
    """Tinker halo mass function (2008).

    Provides fits for multiple overdensity definitions,
    with explicit redshift dependence.
    """

    def __init__(self, cosmology, perturbations, sigma, delta_halo, validadeMassRange):
        """
        Args:
            cosmology: Cosmology object.
            sigma: Variance provider object σ(M).
            delta_halo (float): Halo overdensity definition (Δ).
            validadeMassRange (Callable): Function to check valid mass range.
        """
        super().__init__(cosmology, perturbations, sigma)
        self.delta_halo = delta_halo
        self.validadeMassRange = validadeMassRange
        self.delta_virs = [200, 300, 400, 600, 800, 1200, 1600, 2400, 3200]
        self.A_array = [0.1858659, 0.1995973, 0.2115659, 0.2184113, 0.2480968, 0.2546053, 0.26, 0.26, 0.26]
        self.a_array = [1.466904, 1.521782, 1.559186, 1.614585, 1.869936, 2.128056, 2.301275, 2.529241, 2.661983]
        self.b_array = [2.571104, 2.254217, 2.048674, 1.869559, 1.588649, 1.507134, 1.464374, 1.436827, 1.40521]
        self.c_array = [1.193958, 1.270316, 1.335191, 1.446266, 1.581345, 1.79505, 1.965613, 2.237466, 2.439729]
        self.A_func = spline(self.delta_virs, self.A_array)
        self.a_func = spline(self.delta_virs, self.a_array)
        self.b_func = spline(self.delta_virs, self.b_array)
        self.c_func = spline(self.delta_virs, self.c_array)

    def massFunction(self, lm: float, z: float) -> float:
        """Compute dn/dM following Tinker et al. (2008).

        Args:
            lm (float): Logarithm base 10 of halo mass (M☉/h).
            z (float): Redshift.

        Returns:
            float: Differential number density of halos.
        """
        A_0 = self.A_func(self.delta_halo)
        a_0 = self.a_func(self.delta_halo)
        b_0 = self.b_func(self.delta_halo)
        c_0 = self.c_func(self.delta_halo)
        A = A_0 * (1 + z) ** (-0.14)
        a = a_0 * (1 + z) ** (-0.06)
        alpha = exp(-(0.75 / log(self.delta_halo / 75)) ** 1.2)
        b = b_0 * (1 + z) ** (-alpha)
        c = c_0
        rdmt = self.cosmology.rodm(z)
        step = lm / 20.0
        kmass = 10.0 ** lm
        sgm = self.sigma.fstm(lm)
        self.validadeMassRange(sgm, -0.6, 0.4)
        dsgm_dlgm = dfridr(self.sigma.fstm, lm, step, err=0.0)
        fst = A * ((sgm / b) ** (-a) + 1) * exp(-c / sgm ** 2.0)
        dn_dm = (rdmt / kmass ** 2.0) * fst * abs(dsgm_dlgm) / sgm
        return dn_dm