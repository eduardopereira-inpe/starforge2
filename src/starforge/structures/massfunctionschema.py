from typing import Annotated
from typing import Optional

from pydantic import BaseModel, Field, ConfigDict
from numpy import ndarray

class SigmaCache(BaseModel):
    kmass: Annotated[ndarray, Field(description="Array de massas discretizadas")]
    scale: Annotated[ndarray, Field(description="Array de escalas associadas às massas")]
    zred: Annotated[ndarray, Field(description="Array de redshifts (z) para cálculo")]
    km: Annotated[ndarray, Field(description="Valores de km calculados pela função sigma")]
    sg: Annotated[ndarray, Field(description="Valores sigma correspondentes a km")]
    t_z: Annotated[ndarray, Field(description="Idade do universo em cada redshift")]
    d_c2: Annotated[ndarray, Field(description="Delta crítico corrigido por função de crescimento")]
    rdm2: Annotated[ndarray, Field(description="Densidade de matéria escura em cada redshift")]
    rbr2: Annotated[ndarray, Field(description="Densidade de radiação/bariônica em cada redshift")]
    lmInf: Annotated[float, Field(description="Limite inferior de massa para integração")]
    lmSup: Annotated[float, Field(description="Limite superior de massa para integração")]
    mass_function_type: Annotated[str, Field(description="The Name of Mass Function Used")]
    model_config = ConfigDict(arbitrary_types_allowed=True)



class MassFunctionRange(BaseModel):
    """Schema for the validity range of a mass function."""
    name: str = Field(..., description="Short identifier of the mass function (e.g., 'ST', 'PS').")
    range: Optional[list[float]] = Field(
        None,
        description="Recommended range for ln(σ) or halo mass. None if unrestricted."
    )
    description: str = Field(..., description="Brief explanation of the mass function.")


class MassFunctionRanges(BaseModel):
    """Collection of standard mass function ranges."""

    ST: MassFunctionRange
    TK: MassFunctionRange
    PS: MassFunctionRange
    JK: MassFunctionRange
    W: MassFunctionRange
    WT1: MassFunctionRange
    WT2: MassFunctionRange
    B: MassFunctionRange
    R: MassFunctionRange

    @staticmethod
    def default() -> "MassFunctionRanges":
        """Return a MassFunctionRanges instance with standard values."""
        return MassFunctionRanges(
            ST=MassFunctionRange(
                name="ST",
                range=None,
                description="Sheth–Tormen (1999): Extension of Press–Schechter with ellipsoidal collapse corrections."
            ),
            TK=MassFunctionRange(
                name="TK",
                range=[-1.7, 0.9],
                description="Tinker et al. (2008): Empirical fit to N-body simulations."
            ),
            PS=MassFunctionRange(
                name="PS",
                range=None,
                description="Press–Schechter (1974): First analytical formulation of the halo mass function."
            ),
            JK=MassFunctionRange(
                name="JK",
                range=[-1.2, 1.05],
                description="Jenkins et al. (2001): Empirical fit from large-volume simulations."
            ),
            W=MassFunctionRange(
                name="W",
                range=[10, 15],
                description="Warren et al. (2006): High-resolution simulation–based refinement."
            ),
            WT1=MassFunctionRange(
                name="WT1",
                range=[-0.55, 1.31],
                description="Watson et al. (2013) – parametrization 1: calibrated for Friends-of-Friends halos."
            ),
            WT2=MassFunctionRange(
                name="WT2",
                range=[-0.06, 1.024],
                description="Watson et al. (2013) – parametrization 2: calibrated for Spherical Overdensity halos."
            ),
            B=MassFunctionRange(
                name="B",
                range=None,
                description="Bhattacharya et al. (2011): Sheth–Tormen extension with explicit redshift dependence."
            ),
            R=MassFunctionRange(
                name="R",
                range=[-1.7, 0.9],
                description="Reed et al. (2003): Empirical fit for low-mass halos in simulations."
            ),
        )

    def get(self, key: str) -> Optional[MassFunctionRange]:
        """
        Retrieve a mass function range by its string identifier.

        Args:
            key (str): Short identifier of the mass function (e.g., 'ST', 'PS').

        Returns:
            Optional[MassFunctionRange]: The corresponding MassFunctionRange object,
            or None if the key is not valid.
        """
        return getattr(self, key, None)