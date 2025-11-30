# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Observational Cosmic Star Formation Rate (CSFR).

This module provides access to observational CSFR data from Hopkins (2004).

Data source:
Hopkins, A.M. (2004), ApJ 615, 209.

The dataset is distributed with this package under GPLv3.
"""

from __future__ import annotations
import numpy as np
import importlib.resources as pkg_resources
from typing import Tuple

from starforge import data  # supondo que hopkins_2004.csv esteja em starforge/data/


class ObservationalCSFR:
    """Handler for observational cosmic star formation rate data."""

    def __init__(self, filename: str = "hopkins_2004.dat") -> None:
        """
        Initialize the observational CSFR reader.

        Args:
            filename (str, optional): Name of the CSV file with the data.
                Defaults to 'hopkins_2004.csv'. File must be located in
                the `starforge/data/` package.
        """
        with pkg_resources.open_text(data, filename) as f:
            self.data = np.loadtxt(f, delimiter=",")

    def csf_redshift(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get redshift and CSFR values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of (redshift, CSFR).
        """
        return self.data[:, 0], self.data[:, 1]

    def error_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get asymmetric errors for redshift and CSFR.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - xerr: [lower, upper] errors in redshift
                - yerr: [lower, upper] errors in CSFR
        """
        xerr = np.array([
            self.data[:, 0] - self.data[:, 2],
            self.data[:, 3] - self.data[:, 0],
        ])
        yerr = np.array([
            self.data[:, 1] - self.data[:, 4],
            self.data[:, 5] - self.data[:, 1],
        ])
        return xerr, yerr
