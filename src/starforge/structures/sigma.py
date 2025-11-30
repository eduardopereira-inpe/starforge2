from typing import Optional
from numpy import array, log10
from concurrent.futures import ThreadPoolExecutor, as_completed

from starforge.structures.perturbations import Perturbations
from .sigmachacemanager import SigmaCacheManager
from .sigmafunctions import SigmaFunctions
from .massfunctionschema import SigmaCache


class Sigma:
    """
    Main interface for computing and caching σ(M, z).
    Orchestrates Perturbations for calculation and SigmaCacheManager for persistence.
    """

    def __init__(self, perturbations: Perturbations, sku: Optional[str] = None):
        self.perturbations = perturbations
        self.cosmology = perturbations.cosmological_model
        self._sku = sku
        self.cache_manager = SigmaCacheManager()
        self.sg = None
        self.km = None
        self.scale = None
        self.lmInf, self.lmSup = None, None
        self.mass_function_type = None

    @property
    def sku(self):
        return self._sku

    def parallel_map(self, func, iterable, max_workers: int = 8):
        """Helper for parallel mapping."""
        results = [None] * len(iterable)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(func, val): idx for idx, val in enumerate(iterable)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                results[idx] = future.result()
        return array(results)

    def initialize_sigma_cache(
        self,
        zmax: float,
        mass_function_type: str,
        lmin: float,
        lmax: float,
        dinamicLimits: bool = False,
    ) -> SigmaFunctions:
        """
        Compute (or load) σ(M,z) arrays and integration limits, returning a SigmaFunctions helper.
        """
        if self._sku is None:
            self._sku = f"{self.cosmology.name}_{mass_function_type}_{lmin}_{lmax}_{zmax}_{dinamicLimits}"

        sig_out = self.cache_manager.load(self.sku)
        if sig_out is not None:
            return SigmaFunctions(sig_out)

        # mass range
        mmax = 10 ** lmax
        mmin = 10 ** lmin

        numk = 10000
        ut = 1.0 / 3.0
        kscale = mmax / mmin
        kls = log10(kscale)
        kls1 = kls / numk
        deltaz = zmax / numk

        kmass = array([(10 ** ((i + 1) * kls1)) * mmin for i in range(numk)])
        scale = array([(km / self.cosmology.params.ct2) ** ut for km in kmass])
        zred = array([zmax - i * deltaz for i in range(numk)])

        self.km, self.sg = self.perturbations.sigma(kmass)

        t_z = self.parallel_map(lambda z: self.cosmology.age(z), zred)
        d_c2 = self.parallel_map(
            lambda z: self.cosmology.params.deltac
            / self.perturbations.growth_function(z),
            zred,
        )
        rdm2 = self.parallel_map(lambda z: self.cosmology.rodm(z), zred)
        rbr2 = self.parallel_map(lambda z: self.cosmology.robr(z), zred)

        sig_out = SigmaCache(
            kmass=kmass,
            scale=scale,
            zred=zred,
            km=self.km,
            sg=self.sg,
            t_z=t_z,
            d_c2=d_c2,
            rdm2=rdm2,
            rbr2=rbr2,
            lmInf=lmin,
            lmSup=lmax,
            mass_function_type=mass_function_type,
        )

        self.cache_manager.save(self.sku, sig_out)
        return SigmaFunctions(sig_out)
