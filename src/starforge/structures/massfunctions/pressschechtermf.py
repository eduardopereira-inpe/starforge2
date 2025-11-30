from numpy import exp, sqrt, pi
from starforge.utils.diferencial import dfridr
from .halosmassfunction import HalosMassFunction

class PressSchechterMF(HalosMassFunction):
    """Press-Schechter halo mass function (1974).

    The original analytic model for halo abundances,
    based on Gaussian statistics and spherical collapse.
    """

    def massFunction(self, lm: float, z: float) -> float:
        r"""
        Compute the halo mass function \( \frac{dn}{dM} \) following 
        Press--Schechter (1974).

        The formal expression is:

        \[
        \frac{dn}{dM} = \sqrt{\frac{2}{\pi}} \, 
        \frac{\rho_m(z)}{M^2} 
        \frac{\delta_c}{\sigma(M) D(z)} 
        \exp\!\left[-\frac{\delta_c^2}{2\sigma(M)^2 D(z)^2}\right] 
        \left|\frac{d\ln\sigma}{d\ln M}\right|
        \]

        where:
        - \( \rho_m(z) \): mean matter density at redshift \(z\),
        - \( M \): halo mass,
        - \( \delta_c \): critical density threshold for collapse,
        - \( \sigma(M) \): variance of the smoothed density field at \(z=0\),
        - \( D(z) \): linear growth factor,
        - \( \frac{d\ln\sigma}{d\ln M} \): logarithmic slope of \(\sigma\).

        Note:
        The variance \(\sigma(M)\) is redshift-independent. The time evolution 
        is fully accounted for by the growth factor \(D(z)\).

        Args:
            lm (float): Logarithm base 10 of halo mass \(\log_{10}(M/M_\odot)\).
            z (float): Redshift.

        Returns:
            float: Differential halo number density \(dn/dM\).
        """
        # Linear growth factor at redshift z
        growth = self.perturbations.growth_function(z)

        # Mean matter density at z
        rho_m = self.cosmology.rodm(z)

        # Halo mass
        mass = 10.0 ** lm

        # Variance σ(M) at z=0
        sigma_M = self.sigma.fstm(lm)

        # Derivative dσ/dlnM (numerical, Richardson extrapolation)
        step = lm / 20.0
        d_sigma_dlnM = dfridr(self.sigma.fstm, lm, step, err=0.0)

        # Peak height ν = δ_c / (σ(M) D(z))
        nu = self.cosmology.params.deltac / (sigma_M * growth)

        # Multiplicity function f(σ)
        f_sigma = sqrt(2.0 / pi) * nu * exp(-0.5 * nu**2)

        # Press–Schechter mass function
        dn_dM = (rho_m / mass**2) * f_sigma * abs(d_sigma_dlnM) / sigma_M

        return dn_dM