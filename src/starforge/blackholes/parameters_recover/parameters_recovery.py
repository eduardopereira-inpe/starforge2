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
from typing import List, Tuple
import numpy as np

from ray.actor import ActorClass


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
    ) -> Tuple[np.ndarray, float, ParameterAccretionRate]:
        """
        Calculate the regularization terms, initial condition, and model parameters.

        Args:
            params_array (np.ndarray): Array of parameters. First part corresponds to 
                Gaussian parameters, last seven values are model parameters.
            weight_p (float): Regularization weight for Gaussian parameters.
            weight_theta (float): Regularization weight for model parameters.

        Returns:
            Tuple[np.ndarray, float, ParameterAccretionRate]:
                - Initial condition (np.ndarray)
                - Regularization scalar (float)
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
        regularization = weight_p * np.linalg.norm(normalization_p, 1)

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
        regularization += weight_theta * np.linalg.norm(normalization_model, 1)

        return initial_condition, regularization, accretion_params

    def objective_function(
        self,
        params_array: np.ndarray,
        target_data: np.ndarray,
        weight_p: float = 0.03,
        weight_theta: float = 0.03,
        log_scale: bool = False,
    ) -> float:
        """
        Computes the objective function value for a given set of parameters.

        This function evaluates the loss between the model output and the target data,
        combined with L1 regularization applied to both Gaussian and physical parameters.

        Args:
            params_array (np.ndarray): Optimization parameter vector.
            target_data (np.ndarray): Target distribution for comparison.
            weight_p (float): Regularization weight for Gaussian terms.
            weight_theta (float): Regularization weight for physical parameters.

        Returns:
            float: Loss + regularization.
        """
        try:
            initial_condition, regularization, accretion_params = \
                self._calculate_regularization_terms(
                    params_array, weight_p, weight_theta)
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
        loss = smape(target_data, outputs.n_final)
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

        return float(loss + regularization)

    @ray.method
    def optimizer(
        self,
        bounds: List[Tuple[float, float]],
        target_data: np.ndarray,
        weight_p: float = 0.01,
        weight_theta: float = 0.01,
        n_max: int = 1
    ):
        """
        Optimizes the objective function using the Differential Evolution algorithm.
        This version runs the DE locally inside the Ray actor, ensuring that all
        objective function evaluations remain local and do not incur RPC overhead.
        """
        print("Starting optimization ...")

        def local_objective(params):
            return self.objective_function(
                params,
                target_data,
                weight_p,
                weight_theta
            )

        best = None

        def callback_function(x, f, context):
            context_dict = {
                0: "Minimum detected in annealing process",
                1: "Detection occured in the local search",
                2: "Detection done in the dual annealing process",
            }
            print(
                f"Objective value: {f} "
                f"Context: {context_dict.get(context, 'Unknown')}"
            )

        result = None

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

        for s in range(n_max):
            result = dual_annealing(
                local_objective,
                x0=result.x if result is not None else None,
                bounds=bounds,
                maxiter=2000,                     # maior exploração
                initial_temp=5230.0,              # temperatura maior
                visit=2.62,                       # mais diversidade
                accept=-5.0,                      # aceitação mais permissiva
                restart_temp_ratio=2e-05,
                maxfun=20000,
                minimizer_kwargs={
                    "method": "L-BFGS-B",
                    "options": {
                        "maxiter": 1500,
                        "ftol": 1e-14,
                    },
                },
                callback=callback_function
            )

            if best is None or result.fun < best.fun:
                best = result

            print(f"Stage {s+1}/{n_max}, Best Objective: {best.fun}")

        result = differential_evolution(
            local_objective,     # LOCAL call, no remote RPC
            x0=best.x,
            bounds=bounds,
            strategy="best1bin",
            maxiter=30,
            popsize=20,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            init="latinhypercube",
            disp=True
        )

        if best is None or result.fun < best.fun:
            best = result

        return best


ParametersRecoveryRemote: ActorClass[ParametersRecovery] = ray.remote(
    ParametersRecovery)
