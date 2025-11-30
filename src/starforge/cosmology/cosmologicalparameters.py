from numpy import pi, log, exp, sqrt
from pydantic import BaseModel, Field, model_validator

class CosmologicalParameters(BaseModel):
    """
    Pydantic schema for cosmological parameters.
    """
    omegam: float = Field(0.24, description="Matter density parameter Ω_m")
    omegab: float = Field(0.04, description="Baryonic density parameter Ω_b")
    omegal: float = Field(0.73, description="Dark energy density parameter Ω_Λ")
    h: float = Field(0.7, description="Dimensionless Hubble parameter")
    sigma8: float = Field(0.84, description="Normalization of the power spectrum")
    n_s: float = Field(1.0, description="Primordial power spectrum tilt")
    deltac: float = Field(1.686, description="Critical overdensity for structure collapse.")
    G: float = Field(6.67e-11, description="Gravitational Constant m3 kg-1 s-2")

    # Derived fields (initialized later)
    h2: float = Field(default=None, description="Square of Dimensionless Hubble parameter")
    h2om: float = Field(default=None, description="h^2 * Ω_m")
    h2br: float = Field(default=None, description="h^2 * Ω_b")
    ct0: float = Field(default=None, description="4 * pi")
    ct1: float = Field(default=None, description="ct0 * 2.76e11 / 3")
    ct2: float = Field(default=None, description="ct1 * h2om")
    age_factor: float = Field(default=9.78e+09, description="Age Factor")
    tilt: float = Field(default=None, description="Tilt used for power spectrum normalization")
    anorm: float = Field(default=None, description="Power spectrum normalization constant")
    H0: float = Field(default=None, description="Hubble constant km/s/Mpc")
    anorm: float = Field(default=None, description="Normalization of the power spectrum (sigma_8)")
    gama: float = Field(default=None, description="TODO: Verificar o Significado")
    alfa: float = Field(default=None, description="TODO: Verificar o Significado")
    beta: float = Field(default=None, description="TODO: Verificar o Significado")
    roc0: float = Field(default=None, description="TODO: Verificar o Significado")
    rodm0: float = Field(default=None, description="TODO: Verificar o Significado")
    robr0: float = Field(default=None, description="TODO: Verificar o Significado")
    
    
    

    @model_validator(mode='after')
    def compute_derived_constants(cls, model):
        # h^2 and h^2 * omegam
        model.h2 = model.h ** 2
        model.h2om = model.h2 * model.omegam
        model.h2br =  model.h2 *  model.omegab

        critc = 2.76e+11
        model.roc0 =  critc * model.h2
        model.rodm0 = critc * model.h2om
        model.robr0 = critc * model.h2br

        model.H0 = 100.0 * model.h  

        # Some constants
        model.ct0 = 4.0 * pi
        model.ct1 = model.ct0 * critc / 3.0
        model.ct2 = model.ct1 * model.h2om

        # Tilt depending on omegal
        if model.omegal >= 0.73:
            model.tilt = 1.92
        elif 0.7 <= model.omegal < 0.73:
            model.tilt = 1.915
        else:
            model.tilt = 1.8

        # Power spectrum normalization
        anorm = 1.94e-5 * (model.omegam ** (-0.785 - 0.05 * log(model.omegam)))
        anorm *= exp(-0.95 * (model.tilt - 1.0) - 0.169 * (model.tilt - 1.0)**2) 
        anorm /= 2.0 * (pi ** 2)
        model.anorm = (anorm ** 2) * ((2997.9 / model.h) ** (3.0 + model.tilt))

        gama1 =  model.omegab * (1.0 + sqrt(2.0 * model.h) /  model.omegam)
        gamam =  model.omegam * (model.h ** 2.0) / (gama1)

        model.alfa = 6.4 / gamam /  model.h
        model.beta = 3.0 / gamam /  model.h
        model.gama = 1.7 / gamam /  model.h

        return model

