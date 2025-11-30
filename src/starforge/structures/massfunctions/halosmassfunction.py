from starforge.utils.diferencial import dfridr
from starforge.cosmology import CosmologicalModel
from ..sigmafunctions import SigmaFunctions
from ..perturbations import Perturbations

class HalosMassFunction:
    """Base class for halo mass functions.

    Provides the interface for dark matter halo mass functions,
    which compute the differential abundance of halos as a function
    of halo mass and redshift.

    Attributes:
        cosmology: Cosmology object containing auxiliary methods
            (e.g., mean matter density, growth factor, cosmological parameters).
        sigma: Object that provides the variance of the matter density
            fluctuations σ(M) and the growth function.
    """

    def __init__(self, cosmology: CosmologicalModel, perturbations: Perturbations, sigma: SigmaFunctions):
        """Initialize the halo mass function.

        Args:
            cosmology: Cosmology object with required parameters and methods.
            sigma: Object that provides the density variance σ(M).
        """
        self.cosmology = cosmology
        self.sigma = sigma
        self.perturbations = perturbations

    def massFunction(self, lm: float, z: float) -> float:
        """Compute the halo mass function.

        Args:
            lm (float): Logarithm base 10 of the halo mass (M☉/h).
            z (float): Cosmological redshift.

        Returns:
            float: Differential number density of halos dn/dM.

        Raises:
            Exception: If not implemented in subclass.
        """
        raise Exception("Not implemented Yet")
