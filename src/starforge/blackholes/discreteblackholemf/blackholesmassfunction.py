import logging
from typing import List, Optional
import numpy as np
import tqdm
from starforge.blackholes.accretionrate.blackholesaccretionrate import BlackHolesAccretionRate
from pydantic import BaseModel
import pydantic_numpy.typing as pnd


class SolverOutput(BaseModel):
    z_initial: float = None
    z_final: float = None
    z_grid: pnd.NpNDArray
    ln_mbh_grid: pnd.NpNDArray
    n_initial: pnd.NpNDArray
    u_initial: pnd.NpNDArray
    n_final: pnd.NpNDArray
    u_final: pnd.NpNDArray


class BlackHolesMassFunction:
    """
    Solver for the black hole mass function continuity equation using:
      - grid uniform in x = log10(m / M_ref) (dex)
      - explicit TVD RK2 in redshift (with dt from cosmology)
      - upwind conservative flux discretization
      - optional Lax–Friedrichs stabilization
    """

    def __init__(self, accretionrate: BlackHolesAccretionRate, use_lax_friedrichs: bool = False, logger: logging.Logger = logging.getLogger(__name__)):
        self.accretionrate = accretionrate
        self.use_lax_friedrichs = use_lax_friedrichs
        self.logger = logger

    def f_radiative_efficiency(self, z: float) -> np.ndarray:
        """
        Radiative efficiency eta(z) from accretion rate model.
        """
        eta = self.accretionrate.f_radiative_efficiency(z)
        return eta

    def u_dimensionless(self, z: float, x: np.ndarray) -> np.ndarray:
        """
        Dimensionless velocity \tilde{u}(z, x) is given by:

        \tilde{u}(z, x) = exp[(alpha - 1) * x] * (tau_star / t_u(z)) * exp[-t_u(z) / tau_star]

        where:
        - x = log10(m / M_ref)
        - t_u(z) is the age of the universe at redshift z (in Gyr)
        - tau_star and alpha are model parameters
        """
        age = self.accretionrate.cosmology.age(z)
        tau_star = self.accretionrate.params.tau_par
        alpha = self.accretionrate.params.alpha_par
        u = np.exp((alpha - 1) * x) * (tau_star / age) * \
            np.exp(-age / tau_star)
        return u

    def solver(
        self,
        initial_condition: np.ndarray,
        z_i: float,
        ln_mbh_i: float,
        z_f: float,
        ln_mbh_f: float,
        z_steps: int,
        ln_mbh_steps: int,
        verbose: bool = False,
        data_sampling: Optional[List[tuple[float, float]]] = None
    ) -> SolverOutput:
        """
        Evolve n(m,z) from z_i to z_f on a log10-mass grid.
        initial_condition: array with length ln_mbh_steps (units: number density per dex)
        Returns: SolverOutput pydantic model (keeps arrays as numpy arrays)
        """

        # -------------------------
        # grid definitions
        # -------------------------
        log10_m_grid = np.linspace(
            ln_mbh_i, ln_mbh_f, ln_mbh_steps)  # this is log10(m)
        delta_log10 = log10_m_grid[1] - \
            log10_m_grid[0] if ln_mbh_steps > 1 else 0.0  # dex
        # x = log10(m / M_ref)
        x = log10_m_grid - np.log10(self.accretionrate.params.mbh_par)
        dx = delta_log10  # derivative is with respect to x (dex)

        # -----------------------
        # Theoretical reference quantities
        # -----------------------
        L_theory = self.accretionrate.params.lb_mean_par  # L_sun
        M_theory = self.accretionrate.params.mbh_par  # M_sun
        speed_of_light = self.accretionrate.params.speed_light  # cm/s

        # Physical constants for unit conversion
        L_sun = 3.828e33      # erg/s
        M_sun = 1.989e33      # g
        sec_per_Gyr = 3.1536e16  # s

        # -----------------------
        # Apply trainable multiplicative factors
        # -----------------------

        L_ref = L_theory * L_sun      # erg/s
        M_ref = M_theory * M_sun     # g

        # -----------------------
        # Time scale T0 in Gyr
        # -----------------------
        # T0 = f_ref * (c^2 * M_ref / L_ref)  [s]
        # convert seconds -> Gyr
        T0l = (speed_of_light ** 2.0) * M_ref / L_ref / sec_per_Gyr

        # redshift grid
        if z_steps < 2:
            raise ValueError("z_steps must be >= 2")

        z_grid = np.linspace(z_i, z_f, z_steps)
        dz = (z_f - z_i) / (z_steps - 1)

        # constants
        C_CFL = 0.9

        # initialize solution (assume initial_condition is per dex)
        n_ln = initial_condition.copy()  # n at current step (per dex)
        n_initial = initial_condition.copy()

        # compute initial u for output/diagnostics
        u_initial = self.u_dimensionless(z=z_i, x=x)

        # helper: total number density (integral over dex)
        def total_number(n_field: np.ndarray) -> float:
            # if n is per dex, total number density = sum(n * delta_log10)
            return float(np.sum(n_field) * delta_log10)

        # time-stepping loop over redshift (explicit)
        if verbose:
            iter_z = tqdm.tqdm(
                enumerate(z_grid[:-1]), desc="z steps", leave=False)
        else:
            iter_z = enumerate(z_grid[:-1])

        u = u_initial.copy()
        for k, z in iter_z:
            # dt in Gyr (absolute)
            dt_dz = self.accretionrate.cosmology.dt_dz(z)
            dt = abs(dt_dz * dz) / 1e9  # Gyr
            T0 = T0l * self.f_radiative_efficiency(z)

            # compute dimensionless velocity u(x,z)
            u = self.u_dimensionless(z=z, x=x)
            umax = np.max(np.abs(u))

            # CFL: max|u| * (1/T0) * dt/dx <= C_CFL  => dt <= C_CFL * T0 * dx / max|u|
            dt_cfl = C_CFL * T0 * dx / max(umax, 1e-30)

            if dt > dt_cfl:
                # raise to force user to refine redshift stepping.
                raise RuntimeError(
                    f"CFL condition violated: dt={dt:.3e} Gyr > dt_cfl={dt_cfl:.3e} Gyr at z={z:.3f} (umax={umax:.3e}), T0 = {T0}"
                )

            # store old
            n_old = n_ln.copy()

            # ---------- helpers for flux divergence ----------
            def compute_dF_dx(n_field: np.ndarray) -> np.ndarray:
                # conservative flux F = u * n
                F = u * n_field
                dF = np.zeros_like(n_field)
                # upwind conservative difference: (F_i - F_{i-1}) / dx
                dF[1:] = (F[1:] - F[:-1]) / dx
                dF[0] = 0.0  # zero inflow at left boundary (no incoming flux)
                return dF

            # ---------- TVD RK2 (Heun / modified Euler) ----------
            # Stage 1: explicit Euler
            dF1 = compute_dF_dx(n_old)
            n1 = n_old - (1.0 / T0) * dt * dF1

            # Stage 2: evaluate slope at n1
            dF2 = compute_dF_dx(n1)
            n_new = 0.5 * (n_old + n1 - (1.0 / T0) * dt * dF2)
            # ---------- end RK2 ----------

            # optional Lax-Friedrichs smoothing (stabilization)
            if self.use_lax_friedrichs:
                alpha = umax
                n_lf = n_new.copy()
                # interior points
                if len(n_new) >= 3:
                    n_lf[1:-1] = 0.5 * (n_new[2:] + n_new[:-2]) - \
                        0.5 * alpha * dt / dx * (n_new[2:] - n_new[:-2])
                # boundaries remain as computed (or could be set to zero influx)
                n_new = n_lf

            # update solution
            n_ln = n_new

            # diagnostics (logger)
            try:
                totN = total_number(n_ln)
            except Exception:
                totN = np.nan
            self.logger.debug(
                f"Step {k+1}/{len(z_grid)-1}: z={z:.3f}, dt={dt:.3e} Gyr, total_N={totN:.3e}, min(n)={n_ln.min():.3e}"
            )

        # fim do loop: u_final no último z
        u_final = self.u_dimensionless(z=z_f, x=x)

        # Prepare output (mantendo n em per dex)
        out_put = SolverOutput(
            z_initial=z_i,
            z_final=z_f,
            z_grid=z_grid,
            ln_mbh_grid=log10_m_grid,
            n_initial=n_initial,
            u_initial=u_initial,
            n_final=n_ln,
            u_final=u_final,
        )

        return out_put
