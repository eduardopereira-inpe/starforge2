"""Utility functions for inverse problem experiments.

Contains pure functions used by the inverse problem runner and scripts:
- noise generation
- gaussian initial conditions and gaussian mixtures
- bounds builder for optimization
- simple metrics and plotting helpers
"""
from typing import Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def schechter_initial_condition(ln_mbh: float, phi_star: float, m_star: float, alpha: float, beta: float) -> float:
    """Schechter initial condition used as example (kept as utility)."""
    mbh = np.exp(ln_mbh)
    m_star_exp = np.exp(m_star)
    phi = phi_star * ((mbh / m_star_exp) ** (alpha+1)) * \
        np.exp(1-(mbh / m_star_exp) ** beta)
    return phi


def noise_generator(N: int, mu: float = 0.0, sigma: float = 0.05, rho: float = 0.8, alpha: float = 0.5, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate white, red (AR(1)) and mixed noise arrays."""
    np.random.seed(seed)
    epsilon_white = np.random.normal(mu, sigma, N)
    epsilon_red = np.random.normal(mu, sigma, N)
    delta_red = np.zeros(N)
    for t in range(1, N):
        delta_red[t] = rho * delta_red[t - 1] + epsilon_red[t]
    delta_mixed = alpha * epsilon_white + (1 - alpha) * delta_red
    return epsilon_white, delta_red, delta_mixed


def gaussian_initial_condition(ln_mbh: Sequence[float], mu: float, sigma: float, amplitude: float = 1.0, normalize: bool = True) -> np.ndarray:
    """Return a (optionally normalized) Gaussian evaluated on ln_mbh."""
    ln_mbh = np.asarray(ln_mbh)
    g = np.exp(-0.5 * ((ln_mbh - mu) / sigma) ** 2)
    if normalize:
        area = np.trapezoid(g, ln_mbh)
        if area == 0:
            return np.zeros_like(g)
        g = g / area
    return amplitude * g


def gaussian_sum_from_p(p: np.ndarray, x: np.ndarray, n_gauss: int) -> np.ndarray:
    """Compute linear combination of K Gaussians given parameter vector p = [a.., c.., s..]."""
    if len(p) != 3 * n_gauss:
        raise ValueError(
            f"The parameter vector p must have length 3 × n_gauss = {3 * n_gauss}, but has length {len(p)}.")
    a = p[:n_gauss]
    c = p[n_gauss:2 * n_gauss]
    s = np.abs(p[2 * n_gauss:]) + 1e-6
    x = np.asarray(x)[:, None]
    G = np.exp(-0.5 * ((x - c) / s) ** 2)
    f = G @ a
    return f.ravel()


def triangle_wave(x, period=1, amplitude=1.0):
    """
    Generates a value of a triangle wave using numpy.

    Args:
        x (np.ndarray or float): The input value(s).
        period (float): The period of the triangle wave.

    Returns:
        np.ndarray or float: The corresponding value(s) of the triangle wave.
    """
    x = np.asarray(x)
    x_norm = x % period
    return amplitude * np.where(x_norm < period / 2, (2 / period) * x_norm, 2 - (2 / period) * x_norm)


def triangle_function(x, center=1, amplitude=1.0, width=10.0):
    """
    Generates a value of a triangle function using numpy.

    Args:
        x (np.ndarray or float): The input value(s).
        center (float): The center of the triangle function.
        amplitude (float): The amplitude of the triangle function.

    Returns:
        np.ndarray or float: The corresponding value(s) of the triangle wave.
    """
    x = np.asarray(x)
    x_norm = np.abs(x - center)
    y = np.where(x_norm <= width, amplitude * (1.0 - x_norm / width), 0.0)
    return y


def make_bounds_for_gaussians(x_min: float = 6.0, x_max: float = 15.0,
                              preferred_center_index: int = 2, preferred_center: float = 8.0,
                              preferred_center_radius: float = 1.0,
                              Amax: float = 10.0,
                              Amin: float = 0.0,
                              sigma_min: float = 0.05, sigma_max: float = 5.0,
                              sigma_max_preferred: float = 2.0, K: int = 3) -> list[Tuple[float, float]]:
    """Return bounds list (lower, upper) for p = [a (3), c (3), s (3)]."""

    bounds = []

    a_bounds = [(Amin, Amax)] * K

    bounds.extend(a_bounds)
    c_bounds = []
    for k in range(K):
        if k == preferred_center_index:
            lo = max(preferred_center - preferred_center_radius, x_min)
            hi = min(preferred_center + preferred_center_radius, x_max)
            c_bounds.append((lo, hi))
        else:
            c_bounds.append((x_min, x_max))
    bounds.extend(c_bounds)
    s_bounds = []
    for k in range(K):
        if k == preferred_center_index:
            s_bounds.append((sigma_min, sigma_max_preferred))
        else:
            s_bounds.append((sigma_min, sigma_max))
    bounds.extend(s_bounds)
    return bounds


def l1_regularization(x: np.ndarray) -> float:
    """L1-like regularization computed as sum of absolute gradient magnitudes."""
    return float(np.sum(np.abs(np.gradient(x))))


def add_noise_to_data(data: np.ndarray, mu: float, sigma: float, rho: float, alpha: float, seed: int, noise_type: str) -> np.ndarray:
    """Add selected noise ('white','red','mixed') to input data multiplicatively."""
    N = len(data)
    epsilon_white, delta_red, delta_mixed = noise_generator(
        N, mu, sigma, rho, alpha, seed)
    if noise_type == 'white':
        noise = epsilon_white
    elif noise_type == 'red':
        noise = delta_red
    elif noise_type == 'mixed':
        noise = delta_mixed
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    return data * (1 + noise)


def visualize_initial_conditions(
    initial_condition,
    outputs,
    initial_condition_optimized,
    optimized_outputs,
    image_folder: Path,
    image_name: str = 'initial_conditions_comparison',
):
    """Simple visualization helper used in regularization path loop."""

    image_folder.mkdir(parents=True, exist_ok=True)

    figures = []

    # === FIGURA 1 ===
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(
        optimized_outputs.ln_mbh_grid,
        optimized_outputs.n_final,
        label='Optimized Model',
        color='red',
        linestyle='--'
    )
    ax1.set_yscale('log')
    ax1.set_xlabel('log(M_BH / M_sun)')
    ax1.set_ylabel('Number Density (per dex)')
    ax1.set_title('Optimized Black Hole Mass Function')
    ax1.legend()
    ax1.grid(True)

    fig1.savefig(image_folder / f"{image_name}_a.png")
    figures.append(fig1)

    # === FIGURA 2 ===
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(
        outputs.ln_mbh_grid,
        initial_condition,
        label='Original Initial Condition',
        color='green'
    )
    ax2.plot(
        optimized_outputs.ln_mbh_grid,
        initial_condition_optimized,
        label='Optimized Initial Condition',
        color='orange',
        linestyle='--'
    )
    ax2.set_yscale('log')
    ax2.set_xlabel('log(M_BH / M_sun)')
    ax2.set_ylabel('Number Density (per dex)')
    ax2.set_title('Initial Conditions Comparison')
    ax2.legend()
    ax2.grid(True)

    fig2.savefig(image_folder / f"{image_name}_b.png")
    figures.append(fig2)

    # Retorna as figuras para exibição no notebook/Jupyter
    return figures
