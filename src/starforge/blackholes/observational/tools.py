#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# from starforge.cosmology.lcdmmodel import LCDMModel, CosmologicalParameters

# params = CosmologicalParameters(H0=70.0, omegam=0.3, omegal=0.7, h=0.7,
#                                 rodm0=0.25, robr0=0.05, age_factor=9.78e9)
# cosmo = LCDMModel(params)

# M = -23.0
# m_lim = 22.5
# alpha = -0.5

# zmax = z_max(M, m_lim, alpha, cosmo)
# vmax_val = vmax(M, m_lim, alpha, cosmo)
# print(zmax, vmax_val)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

# Velocidade da luz em km/s
c_kms = 2.99792458e5
FOURPI = 4.0 * np.pi


def k_correction(z, alpha_nu, band_obs='i', band_rest='i', assume_same_band=True):
    """
    K-correction simples para espectro em lei de potência.

    Args:
        z (float): redshift
        alpha_nu (float): índice espectral (f_nu ~ nu^alpha_nu)
        band_obs (str): banda observada
        band_rest (str): banda de referência no repouso
        assume_same_band (bool): se True, ignora diferenças entre bandas

    Returns:
        float: termo K-correction em magnitudes
    """
    if assume_same_band or (band_obs == band_rest):
        return -2.5 * (1.0 + alpha_nu) * np.log10(1.0 + z)
    else:
        # Aqui poderia entrar um tratamento real de diferença entre filtros
        raise NotImplementedError("K-correction entre bandas diferentes não implementado.")


def apparent_magnitude_from_M(M, z, alpha_nu, band_obs='i', band_rest='i',
                              cosmology=None, assume_same_band=True):
    """
    Calcula magnitude aparente a partir da absoluta.

    Args:
        M (float): magnitude absoluta
        z (float): redshift
        alpha_nu (float): índice espectral
        band_obs (str): banda observada
        band_rest (str): banda de referência
        cosmology (LCDMModel): cosmologia
        assume_same_band (bool): se True, usa mesma banda

    Returns:
        float: magnitude aparente
    """
    if cosmology is None:
        raise ValueError("Cosmology object must be provided.")

    D_L_Mpc = cosmology.luminosity_distance(z)  # Mpc
    D_L_pc = D_L_Mpc * 1.0e6
    Kz = k_correction(z, alpha_nu, band_obs, band_rest, assume_same_band=assume_same_band)
    return M + 5.0 * np.log10(D_L_pc / 10.0) + Kz


def absolute_magnitude_from_m(m, z, alpha_nu, band_obs='i', band_rest='i',
                              cosmology=None, assume_same_band=True):
    """
    Calcula magnitude absoluta a partir da aparente.

    Args:
        m (float): magnitude aparente
        z (float): redshift
        alpha_nu (float): índice espectral
        band_obs (str): banda observada
        band_rest (str): banda de referência
        cosmology (LCDMModel): cosmologia
        assume_same_band (bool): se True, usa mesma banda

    Returns:
        float: magnitude absoluta
    """
    if cosmology is None:
        raise ValueError("Cosmology object must be provided.")

    D_L_Mpc = cosmology.luminosity_distance(z)  # Mpc
    D_L_pc = D_L_Mpc * 1.0e6
    Kz = k_correction(z, alpha_nu, band_obs, band_rest, assume_same_band=assume_same_band)
    return m - 5.0 * np.log10(D_L_pc / 10.0) - Kz


def _dVcdz(z, cosmology):
    """
    Elemento de volume comóvel diferencial (sem fator 4pi).

    Args:
        z (float): redshift
        cosmology (LCDMModel)

    Returns:
        float: dV_c/dz (Mpc^3 por sr)
    """
    Dc, _ = quad(cosmology.dr_dz, 0.0, z)  # Mpc
    Hz = cosmology.H(z)  # km/s/Mpc
    return (c_kms * Dc**2 / Hz)


def comoving_volume_between(z1, z2, cosmology, use_fullsky=False):
    """
    Volume comóvel entre dois redshifts.

    Args:
        z1 (float): limite inferior
        z2 (float): limite superior
        cosmology (LCDMModel)
        use_fullsky (bool): se True, retorna 4π sr

    Returns:
        float: volume em Mpc^3
    """
    if z2 <= z1:
        return 0.0
    V2 = cosmology.comoved_volume(z2)  # já full-sky
    V1 = cosmology.comoved_volume(z1)
    V_full = V2 - V1
    return V_full if use_fullsky else V_full / FOURPI


def z_max(M, m_lim, alpha_nu, cosmology, band_obs='i', band_rest='i',
          assume_same_band=True, z_max_search=10.0):
    """
    Redshift máximo em que um objeto com magnitude absoluta M pode ser detectado.

    Args:
        M (float): magnitude absoluta
        m_lim (float): limite de magnitude aparente
        alpha_nu (float): índice espectral
        cosmology (LCDMModel)
        band_obs (str): banda observada
        band_rest (str): banda em repouso
        assume_same_band (bool): se True, ignora diferença entre bandas
        z_max_search (float): limite superior da busca em redshift

    Returns:
        float: z_max
    """
    def m_diff(z):
        m = apparent_magnitude_from_M(M, z, alpha_nu, band_obs=band_obs,
                                      band_rest=band_rest,
                                      cosmology=cosmology,
                                      assume_same_band=assume_same_band)
        return m - m_lim

    try:
        z_max_val = brentq(m_diff, 1e-5, z_max_search)
    except ValueError:
        z_max_val = np.nan
    return z_max_val


def vmax(M, m_lim, alpha_nu, cosmology, z_min=0.0, band_obs='i', band_rest='i',
         assume_same_band=True, z_max_search=10.0):
    """
    Calcula Vmax: volume máximo acessível para um objeto dado M.

    Args:
        M (float): magnitude absoluta
        m_lim (float): limite de magnitude aparente
        alpha_nu (float): índice espectral
        cosmology (LCDMModel)
        z_min (float): redshift mínimo considerado
        band_obs (str): banda observada
        band_rest (str): banda em repouso
        assume_same_band (bool)
        z_max_search (float): redshift máximo da busca

    Returns:
        float: Vmax em Mpc^3/sr
    """
    z_max_val = z_max(M, m_lim, alpha_nu, cosmology,
                      band_obs=band_obs, band_rest=band_rest,
                      assume_same_band=assume_same_band,
                      z_max_search=z_max_search)
    if np.isnan(z_max_val):
        return 0.0
    return comoving_volume_between(z_min, z_max_val, cosmology, use_fullsky=False)


def poisson_error_vmax(vmax_vals, weights=None):
    """
    Erro de Poisson para Vmax (usado em função de luminosidade).

    Args:
        vmax_vals (array): valores de Vmax para cada objeto
        weights (array or None): pesos opcionais

    Returns:
        float: incerteza
    """
    vmax_vals = np.array(vmax_vals)
    if weights is None:
        weights = np.ones_like(vmax_vals)
    else:
        weights = np.array(weights)
    inv_vmax = weights / vmax_vals
    return np.sqrt(np.sum(inv_vmax**2))
