import numpy as np
from starforge.utils.diferencial import locate  # mantém import para compatibilidade, pode ser removido se não usado

from .massfunctionschema import SigmaCache, MassFunctionRanges


class SigmaFunctions:
    """
    Utilities for working with σ(M, z) once cache is available.
    Provides lookups and integration limit helpers.
    """

    def __init__(self, sigma_cache: SigmaCache):
        self.km = sigma_cache.km  # log10(M)
        self.sg = sigma_cache.sg  # σ(M)
        self.lmInf = sigma_cache.lmInf
        self.lmSup = sigma_cache.lmSup
        self.mass_function_type = sigma_cache.mass_function_type
        self.mf_ranges = MassFunctionRanges.default()

    def fstm(self, lm: float) -> float:
        """
        Return sigma(M) for a given log10 halo mass `lm`
        using cached values (linear interpolation).
        """
        if self.sg is None or self.km is None:
            raise ValueError("Sigma cache not initialized.")

        return float(np.interp(lm, self.km, self.sg))

    def massRangeSigma(self, sgmMin: float, sgmMax: float) -> list[float]:
        """
        Return log10 mass range corresponding to sigma range.
        """
        if self.sg is None or self.km is None:
            raise ValueError("Sigma cache not initialized.")

        # σ(M) normalmente decresce com M → invertido para interpolar
        km_rev = self.km[::-1]
        sg_rev = self.sg[::-1]

        lm_min = float(np.interp(sgmMin, sg_rev, km_rev))
        lm_max = float(np.interp(sgmMax, sg_rev, km_rev))

        return [lm_min, lm_max]

    def integration_limits_mass_function(
        self,
        mass_function_type: str,
        lmin: float,
        lmax: float,
        dinamicLimits: bool = False,
    ) -> list[float]:
        """
        Compute integration limits for mass function based on sigma range.
        """
        if not dinamicLimits:
            return [lmin, lmax]

        mf_range = self.mf_ranges.get(mass_function_type)
        if mf_range is None:
            raise ValueError(f"Unknown mass function type '{mass_function_type}'")

        lnsgm = mf_range.range
        if lnsgm is None:
            return [lmin, lmax]

        if mass_function_type == "W":
            return lnsgm

        sgm_min = 10 ** (-lnsgm[0])
        sgm_max = 10 ** (-lnsgm[1])

        return self.massRangeSigma(sgm_min, sgm_max)
