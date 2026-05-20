from .schemes import GridParameters
from .utils import gaussian_sum_from_p

from .metrics import (
    smape,
    # combined_loss,
    # nrmse_by_range,
    # mdsape,
    # loss_median_q95,
    # huber,
    # pearson_corr,
    # mean_abs_log_ratio,
    # relative_rmse,
    # mase,
)

from starforge.cosmology.cosmologicalparameters import CosmologicalParameters
from starforge.cosmology.lcdmmodel import LCDMModel
from starforge.blackholes.discreteblackholemf import (
    BlackHolesMassFunction,
    SolverOutput,

)
from starforge.blackholes.accretionrate.blackholesaccretionrate import (
    BlackHolesAccretionRate,
    ParameterAccretionRate,
)

from scipy.optimize import (
    dual_annealing,
    differential_evolution
)

import ray
from typing import List, Tuple, TypeAlias
import numpy as np

from ray.actor import ActorClass


RegularizationResult: TypeAlias = Tuple[
    np.ndarray,               # initial_condition
    float,                    # total_regularization
    float,                    # R_p
    float,                    # R_theta
    ParameterAccretionRate,   # accretion_params
]


class ParametersRecovery:
    """
    ParametersRecovery is a class designed to handle the optimization of an objective 
    function for a given set of parameters. It provides methods to solve the direct problem, 
    compute regularization terms, evaluate the objective function, and optimize the parameters 
    using the Differential Evolution algorithm.

    Attributes:
        grid_params (GridParameters): The grid parameters used for the solver, which define 
            the initial and final conditions, as well as the resolution of the grid.

    Methods:
        __init__(grid_params: GridParameters):
            Initializes the ObjectiveFunctionRemote instance with the provided grid parameters.

        directly_problem(initial_condition: np.ndarray, accretion_params: ParameterAccretionRate) -> SolverOutput:
            Solves the direct problem using the provided initial condition and parameter set.

        _calculate_regularization_terms(params_array: np.ndarray, weight_p: float, weight_theta: float)
            -> Tuple[np.ndarray, float, ParameterAccretionRate]:
            Calculates the regularization terms, initial condition, and model parameters.

        objective_function(params_array: np.ndarray, target_data: np.ndarray,
                           weight_p: float = 0.03, weight_theta: float = 0.03) -> float:
            Computes the objective function value combining model loss and regularization.

        optimizer(bounds: List[Tuple[float, float]], target_data: np.ndarray,
                  weight_p: float = 0.03, weight_theta: float = 0.03):
            Runs Differential Evolution to optimize the objective function.
    """

    MODEL_PARAM_COUNT = 7
    GAUSS_PARAMS_PER_COMPONENT = 3

    def __init__(self, grid_params: GridParameters):
        """Initialize the ObjectiveFunctionRemote with grid parameters."""
        self.grid_params = grid_params

        # Precompute mass grid once (performance improvement)
        self.mbh_grid = np.linspace(
            grid_params.mbh_i, grid_params.mbh_f, grid_params.mbh_steps
        )

        # Cosmology objects are static w.r.t. optimization parameters
        self.lcdm = LCDMModel(CosmologicalParameters())

    def directly_problem(
        self, initial_condition: np.ndarray, accretion_params: ParameterAccretionRate
    ) -> SolverOutput:
        """
        Solves the direct problem using the provided initial conditions and grid parameters.

        Args:
            initial_condition (np.ndarray): Initial state for the solver.
            accretion_params (ParameterAccretionRate): Physical parameters for accretion.

        Returns:
            SolverOutput: Output of the mass function solver.
        """

        accretion = BlackHolesAccretionRate(
            cosmology=self.lcdm,
            params=accretion_params
        )

        solver = BlackHolesMassFunction(accretion)

        outputs = solver.solver(
            initial_condition,
            self.grid_params.z_i,
            self.grid_params.mbh_i,
            self.grid_params.z_f,
            self.grid_params.mbh_f,
            self.grid_params.z_steps,
            self.grid_params.mbh_steps,
            verbose=False
        )
        return outputs

    def _calculate_regularization_terms(
        self,
        params_array: np.ndarray,
        weight_p: float,
        weight_theta: float
    ) -> RegularizationResult:
        """
        Calculate the regularization terms, initial condition, and model parameters.

        Args:
            params_array (np.ndarray): Array of parameters. First part corresponds to 
                Gaussian parameters, last seven values are model parameters.
            weight_p (float): Regularization weight for Gaussian parameters.
            weight_theta (float): Regularization weight for model parameters.

        Returns:
            RegularizationResult:
                - Initial condition (np.ndarray)
                - Regularization scalar (float)
                - R_p (float)
                - R_theta (float)
                - ParameterAccretionRate instance
        """

        n = self.MODEL_PARAM_COUNT
        n_gauss = (params_array.shape[0] -
                   n) // self.GAUSS_PARAMS_PER_COMPONENT

        # Build initial condition using Gaussian expansion
        initial_condition = gaussian_sum_from_p(
            # Bloco 1 -- Parametros das Gaussianas
            params_array[:-n],
            self.mbh_grid,
            n_gauss=n_gauss
        )
        # --- Regularization terms ------------------------------------------------------

        # --------------------------------------------------------------------------------
        # OBS: Normalization is done to avoid scale issues between parameters
        # --------------------------------------------------------------------------------

        # --- Avoid log(0) issues --------------------------------------------------------
        epsilon = 1e-10

        if (params_array[:-n] < 0).any():
            raise ValueError("Gaussian parameters contain negative values.")

        # --- Regularization on Gaussian parameters ---------------------------------------
        # --- Normalize Gaussian parameters for consistent regularization -----------------

        normalization_p = np.log10(params_array[:-n] + epsilon)
        normalization_p = (normalization_p - normalization_p.mean()
                           ) / (normalization_p.std() + epsilon)

        # Regularization term on Gaussian parameters
        R_p = np.linalg.norm(normalization_p, 1)

        # Build physical model parameters
        accretion_params = ParameterAccretionRate(
            # Bloco 2 -- Parametros da Luminosidade
            mbh_par=params_array[-n] * 1e11,
            tau_par=params_array[-n+1] * 1e9,
            alpha_par=params_array[-n+2],

            # Bloco 3 -- Parametros da eficiencia Radiativa
            eta=params_array[-n+3],
            t_q=params_array[-n+4] * 1e9,
            b_1=params_array[-n+5],
            b_2=params_array[-n+6]
        )

        # --- Regularization on model parameters --------------------------------------------
        # --- Normalize model parameters for consistent regularization ----------------------
        normalization_model = np.log10(params_array[-n:] + epsilon)
        normalization_model = (normalization_model - normalization_model.mean()
                               ) / (normalization_model.std() + epsilon)

        # Regularization on model parameters
        R_theta = np.linalg.norm(normalization_model, 1)

        regularization = (
            weight_p * R_p +
            weight_theta * R_theta
        )

        return (
            initial_condition,
            regularization,
            R_p,
            R_theta,
            accretion_params
        )

    def objective_function(
        self,
        params_array: np.ndarray,
        target_data: np.ndarray,
        weight_p: float = 0.03,
        weight_theta: float = 0.03,
        log_scale: bool = False,
        return_components: bool = False,
        loss_type: str = "l2"
    ) -> float:
        """
        Computes the objective function value for a given set of parameters.

        This function evaluates the loss between the model output and the 
        target data,
        combined with L1 regularization applied to both Gaussian and physical 
        parameters.

        Args:
            params_array (np.ndarray): Optimization parameter vector.
            target_data (np.ndarray): Target distribution for comparison.
            weight_p (float): Regularization weight for Gaussian terms.
            weight_theta (float): Regularization weight for physical 
            parameters.

        Returns:
            float: Loss + regularization.
        """
        try:
            (
                initial_condition,
                regularization,
                R_p,
                R_theta,
                accretion_params
            ) = self._calculate_regularization_terms(
                params_array,
                weight_p,
                weight_theta
            )
        except Exception as e:
            # print("Error in regularization calculation:", e)
            return np.inf

        try:
            outputs = self.directly_problem(
                initial_condition, accretion_params)
        except Exception as e:
            # print("Error in direct problem solver:", e)
            return np.inf

        if log_scale:
            target_data = np.log10(target_data + 1e-15)
            outputs.n_final = np.log10(outputs.n_final + 1e-15)

        # loss = np.linalg.norm(outputs.n_final - target_data, 2)

        # -- Melhor Resultado Ate Agora --
        if loss_type == "smape":
            loss = smape(target_data, outputs.n_final)
        else:
            loss = np.linalg.norm(
                outputs.n_final - target_data,
                ord=2
            )
        # ---------------------------------

        # loss = combined_loss(target_data, outputs.n_final)
        # loss = nrmse_by_range(target_data, outputs.n_final)

        # -- Resultado Razoavel, mas ruim para altas massas --
        # loss = mdsape(target_data, outputs.n_final)
        # ----------------------------------------------------

        # -- Resultado Razoavel, mas ruim para altas massas --
        # loss = loss_median_q95(target_data, outputs.n_final)
        # ----------------------------------------------------

        # -- Deu um bom ajusta para o dado observado, mas muito ruim para a condição inicial --
        # loss = huber(target_data, outputs.n_final, delta=2e-6)
        # --------------------------------------------------------------------------------------

        # -- Pior resultado ate agora --
        # loss = pearson_corr(target_data, outputs.n_final)
        # --------------------------------------------------------------------------------------

        # -- Resultado Razoavel, ruim para altas massas --
        # loss = mean_abs_log_ratio(target_data, outputs.n_final)
        # ----------------------------------------------------

        # -- Resultado Razoavel, muito bom para altas massas. ruim para baixas massas --
        # loss = relative_rmse(target_data, outputs.n_final)
        # --------------------------------------------------------------------------------

        # -- Resultado Razoavel, muito bom para altas massas. ruim para baixas massas --
        # loss = mase(target_data, outputs.n_final)
        # --------------------------------------------------------------------------------

        objective = float(loss + regularization)

        if return_components:
            return {
                "objective": objective,
                "misfit": float(loss),
                "R_p": float(R_p),
                "R_theta": float(R_theta),
            }

        return objective

    def l_surface_scan(
        self,
        bounds,
        target_data,
        alpha_p_values,
        alpha_theta_values,
    ):
        """
        Compute the L-surface over a grid of
        (alpha_p, alpha_theta).
        """

        results = []

        for alpha_p in alpha_p_values:
            for alpha_theta in alpha_theta_values:

                print(
                    f"Running alpha_p={alpha_p:.2e}, "
                    f"alpha_theta={alpha_theta:.2e}"
                )

                result = self.optimizer(
                    bounds=bounds,
                    target_data=target_data,
                    weight_p=alpha_p,
                    weight_theta=alpha_theta,
                    n_max=1
                )

                diagnostics = self.objective_function(
                    result.x,
                    target_data,
                    weight_p=alpha_p,
                    weight_theta=alpha_theta,
                    return_components=True
                )

                results.append({
                    "alpha_p": alpha_p,
                    "alpha_theta": alpha_theta,
                    "objective": diagnostics["objective"],
                    "misfit": diagnostics["misfit"],
                    "R_p": diagnostics["R_p"],
                    "R_theta": diagnostics["R_theta"],
                    "x": result.x,
                })

        return results

    # ============================= IMPORTANT COMMENT, do not remove ============================================================
    # Dual Annealing – Parameter Tuning Reference (When results are good but can improve)
    # -----------------------------------------------------------------------------
    # Parameter        | Action                        | Effect
    # -----------------|-------------------------------|--------------------------------------------
    # maxiter          | Increase (e.g., 1000 → 2000)  | Deeper global exploration; improves chance of escaping shallow minima
    # initial_temp     | Increase (e.g., 5230 → 8000)  | Expands early exploration; higher acceptance of uphill moves
    # visit            | Slightly increase (3.6 → 3.9) | Enables broader jumps; increases search diversity
    # accept           | Make less negative (-2 → -1)  | More permissive acceptance; may reveal better basins
    # maxfun           | Increase (10000 → 20000)      | Allows additional evaluations; useful when nearing plateau
    # minimizer_kwargs.maxiter | Increase (e.g., 500 → 1500) | Stronger local refinement; improves final precision
    # minimizer_kwargs.ftol    | Decrease (e.g., 1e-9 → 1e-14) | Higher local accuracy; reduces fine-scale variability
    # seed             | Control (e.g., seed=s)        | Ensures reproducible variation across runs
    # Best-to-Next x0  | Use best.x as next x0         | Layered improvement; builds on previous best solution
    # ===========================================================================================================================

    @ray.method
    def optimizer(
        self,
        bounds: List[Tuple[float, float]],
        target_data: np.ndarray,
        weight_p: float = 0.1,
        weight_theta: float = 0.1,
        n_max: int = 1,
        x0: np.ndarray | None = None,
        fast_mode: bool = True,
        use_de: bool = False,
    ):
        """
        Optimizes the objective function.

        Parameters
        ----------
        fast_mode : bool
            If True, uses lightweight optimization settings suitable
            for L-surface scans.

        use_de : bool
            If True, runs Differential Evolution after Dual Annealing.
            Usually unnecessary for coarse L-surface computation.
        """

        print("Starting optimization ...")

        # ============================================================
        # FAST MODE CONFIGURATION
        # ============================================================

        if fast_mode:

            # --- Dual Annealing ---
            da_maxiter = 40
            da_initial_temp = 500.0
            da_visit = 2.2
            da_accept = -10.0
            da_restart_temp_ratio = 1e-4
            da_maxfun = 1500

            # --- Local Minimizer ---
            lbfgs_maxiter = 80
            lbfgs_ftol = 1e-6

            # --- Differential Evolution ---
            de_maxiter = 5
            de_popsize = 5

        # ============================================================
        # FULL / PRODUCTION MODE
        # ============================================================

        else:

            # --- Dual Annealing ---
            da_maxiter = 2000
            da_initial_temp = 5230.0
            da_visit = 2.62
            da_accept = -5.0
            da_restart_temp_ratio = 2e-5
            da_maxfun = 20000

            # --- Local Minimizer ---
            lbfgs_maxiter = 1500
            lbfgs_ftol = 1e-14

            # --- Differential Evolution ---
            de_maxiter = 30
            de_popsize = 20

        # ============================================================
        # OBJECTIVE CACHE
        # ============================================================

        cache = {}

        def local_objective(params):

            key = tuple(np.round(params, 8))

            if key in cache:
                return cache[key]

            value = self.objective_function(
                params,
                target_data,
                weight_p,
                weight_theta
            )

            cache[key] = value

            return value

        # ============================================================
        # CALLBACK
        # ============================================================

        def callback_function(x, f, context):

            context_dict = {
                0: "Minimum detected in annealing process",
                1: "Detection occurred in local search",
                2: "Detection done in dual annealing process",
            }

            print(
                f"Objective value: {f:.6e} | "
                f"Context: {context_dict.get(context, 'Unknown')}"
            )

        # ============================================================
        # MULTI-STAGE DUAL ANNEALING
        # ============================================================

        best = None
        result = None

        current_x0 = x0

        for s in range(n_max):

            result = dual_annealing(
                local_objective,
                bounds=bounds,
                x0=current_x0,
                maxiter=da_maxiter,
                initial_temp=da_initial_temp,
                visit=da_visit,
                accept=da_accept,
                restart_temp_ratio=da_restart_temp_ratio,
                maxfun=da_maxfun,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "options": {
                        "maxiter": lbfgs_maxiter,
                        "ftol": lbfgs_ftol,
                    },
                },
                callback=callback_function,
                no_local_search=False,
            )

            if best is None or result.fun < best.fun:
                best = result

            current_x0 = result.x.copy()

            print(
                f"Stage {s+1}/{n_max} | "
                f"Best Objective: {best.fun:.6e}"
            )

        # ============================================================
        # OPTIONAL DIFFERENTIAL EVOLUTION
        # ============================================================

        if use_de:

            print("Starting Differential Evolution refinement ...")

            result = differential_evolution(
                local_objective,
                bounds=bounds,
                strategy="best1bin",
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=0.01,
                mutation=(0.5, 1.0),
                recombination=0.7,
                init="latinhypercube",
                polish=True,
                disp=True,
                x0=best.x,
            )

            if result.fun < best.fun:
                best = result

        print(
            f"Optimization completed | "
            f"Final Objective: {best.fun:.6e}"
        )

        return best


ParametersRecoveryRemote: ActorClass[ParametersRecovery] = ray.remote(
    ParametersRecovery)
