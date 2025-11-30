#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Setup script for the Sentinel Data Handler package.

This script uses setuptools to package the Sentinel Data Handler, which
provides functionality for handling and processing Sentinel-2 data.
"""

import os
from setuptools import setup, find_packages


def read(filename):
    """Read the content of a file.

    Args:
        filename (str): The name of the file to read.

    Returns:
        str: The content of the file as a string.
    """
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="Star Forge",
    license="Apache License 2.0",
    version='0.1.0',
    author='Eduardo S. Pereira',
    author_email='eduardo.pereira@inpe.br',
    packages=find_packages("src"),  # Finds packages in the 'src' directory
    package_dir={"": "src"},  # Sets the root package directory to 'src'
    description="Star Forge",  # Short description of the package
    # Detailed description from README file
    long_description=read("README.md"),
    # Specifies the format of the long description
    long_description_content_type="text/markdown",

    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'pandas',
        'pydantic',
        'pydantic_numpy',
        'pydal',
        'joblib',
        'tqdm',
        'optimparallel',
        'tensorflow[and-cuda]',
        "scikit-learn",
        'yaspin',
        'seaborn',
        'ray[default]',
        'ipywidgets',
    ],

)
