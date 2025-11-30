from typing import Optional
import numpy as np
from pydantic import BaseModel, model_validator


class Parameters(BaseModel):
    """Physical and noise parameters used by the inverse problem."""
    eta: float = 0.1
    lb_mean_par: float = 7.96e13
    mbh_par: float = 2.19e11
    alpha_par: float = 2.71e-1
    tau_par: float = 4.81e9
    t_q: float = 1e9
    b_1: float = 0.1
    b_2: float = 0.1
    mu: float = 8.0
    sigma: float = 1.5
    amplitude: float = 1e-3
    alpha: float = -1.5
    phi_star: float = 1e-5
    m_star: float = np.log(1e8)
    mu_noise: float = 0.0
    sigma_noise: float = 0.01
    rho_noise: float = 0.8
    alpha_noise: float = 0.5


class GridParameters(BaseModel):
    """Grid definition for mass and redshift discretization."""
    mbh_i: float = 6
    mbh_f: float = 15
    mbh_steps: int = 50
    z_i: float = 5.0
    z_f: float = 0.0
    z_steps: Optional[int] = None
    delta_z: float = 0.1

    @model_validator(mode='after')
    def set_z_steps(self):
        if self.z_steps is None:
            self.z_steps = self.mbh_steps * 100
        return self
