"""
Vmax pipeline adapted for SDSS DR16Q-derived catalog.

Assumptions and inputs:
- Catalog is a FITS or CSV table with columns (as in your derived catalog):
  - Z_DR16Q (redshift)
  - M_I (absolute i-band magnitude corrected for extinction) OR PSFMAG (array or PSFMAG_I)
  - EBV (E(B-V))
  - LOGMBH (log10 BH mass) optional
  - PSFMAG (if M_I absent) -- we expect PSFMAG_i available as PSFMAG[:,3] or PSFMAG_I column

- Default selection/parameters:
  - band_obs = 'i'
  - band_rest = 'i'
  - alpha_nu = -0.5
  - i_lim = 19.1 (SDSS uniform quasar limit; adjust per sub-sample)
  - survey_area_deg2 = 6248.0 (SDSS DR7 uniform sample as example) -- change to your footprint area

Outputs:
- Table with z_max, V_max for each object (for a chosen redshift bin), and binned BHMF table Phi(M,z).

Dependencies: astropy, numpy, scipy

Usage:
- Adjust `catalog_path` and `output_prefix` and run in Python environment.

"""

import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import brentq
from scipy.integrate import quad

# Constants
c_kms = 299792.458
FOURPI = 4.0 * np.pi

# SDSS extinction coefficients (Schlafly & Finkbeiner 2011)
SDSS_A_OVER_EBV = {
    'u': 4.239,
    'g': 3.303,
    'r': 2.285,
    'i': 1.698,
    'z': 1.263,
}

# Effective wavelengths (Angstrom)
EFFECTIVE_WAVELENGTHS = {
    'u': 3543.0,
    'g': 4770.0,
    'r': 6231.0,
    'i': 7625.0,
    'z': 9134.0,
    'uv1450': 1450.0,
}

# -----------------------
# K-correction & mags
# -----------------------
def k_correction(z, alpha_nu, band_obs='i', band_rest='i', assume_same_band=True):
    z = np.asarray(z)
    K = -2.5 * (1.0 + alpha_nu) * np.log10(1.0 + z)
    if (not assume_same_band) or (band_obs != band_rest):
        lambda_obs = EFFECTIVE_WAVELENGTHS[band_obs]
        lambda_rest = EFFECTIVE_WAVELENGTHS[band_rest]
        K = K + 2.5 * alpha_nu * np.log10(lambda_obs / lambda_rest)
    return K if K.shape != () else float(K)


def apparent_mag_from_M(M, z, alpha_nu, band_obs='i', band_rest='i', cosmology=cosmo, assume_same_band=True):
    z = np.asarray(z)
    # luminosity distance in Mpc
    D_L_Mpc = cosmology.luminosity_distance(z).value
    D_L_pc = D_L_Mpc * 1.0e6
    Kz = k_correction(z, alpha_nu, band_obs, band_rest, assume_same_band=assume_same_band)
    m = M + 5.0 * np.log10(D_L_pc / 10.0) + Kz
    return m


def compute_zmax_from_M(M, m_lim, alpha_nu, band_obs, band_rest, cosmology=cosmo,
                        z_min=1e-8, z_max_upper=7.0, assume_same_band=True, tol=1e-6):
    def f(z):
        return apparent_mag_from_M(M, z, alpha_nu, band_obs, band_rest, cosmology, assume_same_band) - m_lim

    f_min = f(z_min)
    f_max = f(z_max_upper)

    if f_min >= 0:
        return float(z_min)
    if f_max <= 0:
        return float(z_max_upper)

    z_root = brentq(f, z_min, z_max_upper, xtol=tol)
    return float(z_root)


def _dVcdz(z, cosmology=cosmo):
    Dc = cosmology.comoving_distance(z).value
    Hz = cosmology.H(z).value
    return (c_kms * Dc**2 / Hz)


def comoving_volume_between(z1, z2, cosmology=cosmo):
    if z2 <= z1:
        return 0.0
    try:
        V2 = cosmology.comoving_volume(z2).value
        V1 = cosmology.comoving_volume(z1).value
        V_full = V2 - V1
        return V_full / FOURPI  # return per steradian
    except Exception:
        I, _ = quad(lambda zz: _dVcdz(zz, cosmology), z1, z2, epsabs=1e-6, epsrel=1e-6)
        return I


def Vmax_for_object(M, m_lim, alpha_nu, band_obs, band_rest, cosmology,
                    Omega_survey_sr, z_bin_low=0.0, z_bin_high=None,
                    assume_same_band=True, z_upper_global=7.0):
    if z_bin_high is None:
        z_bin_high = z_upper_global

    z_detect_max = compute_zmax_from_M(M, m_lim, alpha_nu, band_obs, band_rest, cosmology,
                                       z_min=1e-8, z_max_upper=z_upper_global, assume_same_band=assume_same_band)

    z_start = max(z_bin_low, 1e-8)
    z_end = min(z_bin_high, z_detect_max)
    if z_end <= z_start:
        return 0.0

    V_per_sr = comoving_volume_between(z_start, z_end, cosmology)
    V = V_per_sr * Omega_survey_sr
    return float(V)

# -----------------------
# Pipeline wrapper
# -----------------------

def run_vmax_pipeline(catalog_path, output_prefix,
                      band_obs='i', band_rest='i', alpha_nu=-0.5,
                      i_lim=19.1, survey_area_deg2=6248.0,
                      z_bins=None, mass_bins=None,
                      completeness_column=None):
    """
    Read catalog, compute z_max and V_max per object, produce BHMF table.

    catalog_path: path to FITS/CSV readable by astropy.table.Table.read
    output_prefix: prefix for outputs (CSV/TXT)
    """
    # read catalog
    cat = Table.read(catalog_path)

    # area
    Omega_sr = survey_area_deg2 * (np.pi/180.0)**2

    # extract redshift and absolute magnitude M_I if available
    if 'M_I' in cat.colnames:
        M_i = np.array(cat['M_I'], dtype=float)
    else:
        # try to compute from PSFMAG_i and EBV
        if 'PSFMAG' in cat.colnames:
            # PSFMAG may be stored as array per row; try PSFMAG_I or take column 3
            try:
                psfmag_i = np.array([row['PSFMAG'][3] for row in cat])
            except Exception:
                raise RuntimeError("PSFMAG present but could not extract i-band. Provide M_I instead.")
        elif 'PSFMAG_I' in cat.colnames:
            psfmag_i = np.array(cat['PSFMAG_I'], dtype=float)
        else:
            raise RuntimeError('Catalog must contain M_I or PSFMAG/PSFMAG_I to compute absolute magnitudes.')

        # correct for Galactic extinction using EBV
        ebv = np.array(cat['EBV']) if 'EBV' in cat.colnames else np.zeros_like(psfmag_i)
        A_i = SDSS_A_OVER_EBV['i'] * ebv
        psfmag_i_corr = psfmag_i - A_i

        z_obs = np.array(cat['Z_DR16Q'], dtype=float)
        M_i = apparent_mag_to_absolute(psfmag_i_corr, z_obs, alpha_nu, band_obs, band_rest, cosmo)

    # helper to compute absolute from apparent
    def apparent_mag_to_absolute(m_arr, z_arr, alpha_nu, band_obs, band_rest, cosmology):
        M_arr = []
        for m, z in zip(m_arr, z_arr):
            M_arr.append( absolute_mag_from_apparent(m, z, alpha_nu, band_obs, band_rest, cosmology) )
        return np.array(M_arr)

    # local functions for absolute/m conversions (using the earlier definitions)
    def absolute_mag_from_apparent(m, z, alpha_nu, band_obs, band_rest, cosmology):
        D_L_Mpc = cosmology.luminosity_distance(z).value
        D_L_pc = D_L_Mpc * 1e6
        Kz = k_correction(z, alpha_nu, band_obs, band_rest, assume_same_band=True)
        M = m - 5.0 * np.log10(D_L_pc / 10.0) - Kz
        return float(M)

    # now compute z_max and V_max for each object for a given binning
    z_obs = np.array(cat['Z_DR16Q'], dtype=float)
    Nobj = len(cat)

    # default bins
    if z_bins is None:
        z_bins = np.arange(0.3, 3.1, 0.2)
    if mass_bins is None:
        mass_bins = np.arange(7.0, 10.5, 0.5)

    # compute z_max per object (global detectability)
    zmax_list = np.zeros(Nobj, dtype=float)
    for i in range(Nobj):
        zmax_list[i] = compute_zmax_from_M(M_i[i], i_lim, alpha_nu, band_obs, band_rest, cosmo,
                                           z_min=max(1e-8, z_obs[i]), z_max_upper=7.0, assume_same_band=True)

    # For each redshift bin and mass bin compute Vmax per object and sum 1/V
    results = []
    logM = np.array(cat['LOGMBH']) if 'LOGMBH' in cat.colnames else None

    for iz in range(len(z_bins)-1):
        zlow = z_bins[iz]
        zhigh = z_bins[iz+1]
        for im in range(len(mass_bins)-1):
            mlow = mass_bins[im]
            mhigh = mass_bins[im+1]
            # select objects in observed z bin
            sel = (z_obs >= zlow) & (z_obs < zhigh)
            if logM is not None:
                sel = sel & (logM >= mlow) & (logM < mhigh)

            idx = np.where(sel)[0]
            if len(idx) == 0:
                results.append({'zbin':(zlow,zhigh),'mbin':(mlow,mhigh),'N':0,'Phi':0.0,'err':0.0})
                continue

            V_list = np.zeros(len(idx), dtype=float)
            for k, iobj in enumerate(idx):
                # compute Vmax inside this redshift bin using zmax computed earlier
                z_start = max(zlow, 1e-8)
                z_end = min(zhigh, zmax_list[iobj])
                if z_end <= z_start:
                    V_list[k] = 0.0
                else:
                    V_per_sr = comoving_volume_between(z_start, z_end, cosmology=cosmo)
                    V_list[k] = V_per_sr * Omega_sr

            mask = V_list > 0
            if not np.any(mask):
                results.append({'zbin':(zlow,zhigh),'mbin':(mlow,mhigh),'N':0,'Phi':0.0,'err':0.0})
                continue

            # completeness
            if completeness_column is not None and completeness_column in cat.colnames:
                comp = np.array(cat[completeness_column], dtype=float)[idx][mask]
            else:
                comp = np.ones(mask.sum())

            weights = 1.0 / (V_list[mask] * comp)
            delta_logM = (mhigh - mlow)
            Phi = np.sum(weights) / delta_logM
            err = np.sqrt(np.sum(weights**2)) / delta_logM
            results.append({'zbin':(zlow,zhigh),'mbin':(mlow,mhigh),'N':mask.sum(),'Phi':Phi,'err':err})

    # save results to table
    import csv
    out_csv = output_prefix + '_bhmf_vmax.csv'
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z_low','z_high','M_low','M_high','N','Phi_Mpc3_dex','err'])
        for r in results:
            zlow,zhigh = r['zbin']
            mlow,mhigh = r['mbin']
            writer.writerow([zlow,zhigh,mlow,mhigh,r['N'],r['Phi'],r['err']])

    # also save per-object zmax and (Vmax for full binning top-level) for inspection
    perobj_tbl = Table()
    perobj_tbl['SDSS_NAME'] = cat['SDSS_NAME'] if 'SDSS_NAME' in cat.colnames else np.arange(Nobj)
    perobj_tbl['Z_DR16Q'] = z_obs
    perobj_tbl['M_I'] = M_i
    perobj_tbl['z_max'] = zmax_list
    perobj_tbl.write(output_prefix + '_per_object.fits', overwrite=True)

    print('Saved BHMF table to', out_csv)
    print('Saved per-object table to', output_prefix + '_per_object.fits')
    return results

# End of pipeline file
