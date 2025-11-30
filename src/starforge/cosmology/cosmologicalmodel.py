from typing import Optional
from .cosmologicalparameters import CosmologicalParameters

class CosmologicalModel:
    """
    Abstract class for the cosmological background evolution.

    This interface defines the fundamental quantities that describe the
    homogeneous and isotropic universe, such as the Hubble parameter, 
    comoving distances, and the age of the universe.
    """
    params: CosmologicalParameters
    _name: Optional[str] = None

    @property
    def name(self):
        if self._name is None:
            raise Exception("You must define the Name of cosmological model")
        return self._name


    def H(self, z):
        """
        Compute the Hubble parameter at a given redshift.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Hubble parameter H(z) in km/s/Mpc.

        Notes:
            H(z) defines the expansion rate of the universe at redshift z.
        """
        raise NotImplementedError

    def age(self, z):
        """
        Compute the age of the universe at a given redshift.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Age of the universe in Gyr.

        Notes:
            The age is obtained by integrating 1/H(z) over redshift.
        """
        raise NotImplementedError

    def dr_dz(self, z):
        """
        Compute the differential comoving distance per unit redshift.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: dr/dz in Mpc.

        Notes:
            Used to compute comoving distances and volumes.
        """
        raise NotImplementedError

    def dV_dz(self, z):
        """
        Compute the comoving volume element per unit redshift.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Differential comoving volume in Mpc^3 per steradian.

        Notes:
            Typically used to convert number densities to number counts.
        """
        raise NotImplementedError

    def density_parameters(self, z):
        """
        Return the density parameters of the universe at redshift z.

        Args:
            z (float or array-like): Redshift.

        Returns:
            dict: Dictionary with keys:
                - 'Omega_m': Matter density parameter
                - 'Omega_lambda': Dark energy density parameter
                - 'Omega_r': Radiation density parameter
                - 'Omega_k': Curvature density parameter

        Notes:
            The sum of all Omega's should satisfy the Friedmann equation.
        """
        raise NotImplementedError
    
    def rodmz(self, z):
        return self.rodm(z)
    
    def luminosity_distance(self, z):
        """
        Compute the luminosity distance at a given redshift.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Luminosity distance D_L in Mpc.

        Notes:
            D_L is related to the comoving distance and scale factor.
        """
        raise NotImplementedError
    
    def comoved_volume(self, z):
        """
        Compute the comoving volume out to redshift z.

        Args:
            z (float or array-like): Redshift.

        Returns:
            float or array-like: Comoving volume in Mpc^3.

        Notes:
            Used to compute total volumes for surveys.
        """
        raise NotImplementedError

