#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import kurtosis, skew, scoreatpercentile, iqr, variation

def mean(vol):
    """Compute statistical mean feature.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            --> vol: continuos imaging intensity distribution

    Returns:
        float: Statistical mean feature
    """
    # INITIALIZATION
    X = vol[~np.isnan(vol[:])]
    return np.mean(X)

def extract_all(vol, intensity=None):
    """Compute StatsFeatures.

    - vol: 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
        --> vol: continuos imaging intensity distribution
    - intensity (optional): If 'arbitrary', some feature will not be computed.
        If 'definite', all feature will be computed. If not present as an argument,
        all features will be computed. Here, 'filter' is the same as
        'arbitrary'.
    """

    # PRELIMINARY

    if intensity is None:
        definite = True
    elif intensity == 'arbitrary':
        definite = False
    elif intensity == 'definite':
        definite = True
    elif intensity == 'filter':
        definite = False
    else:
        raise ValueError('Second argument must either be "arbitrary" or \
                         "definite" or "filter"')

    # INITIALIZATION

    X = vol[~np.isnan(vol[:])]

    # Initialization of final structure (Dictionary) containing all features.
    stats = {'Fstat_mean': [],
             'Fstat_var': [],
             'Fstat_skew': [],
             'Fstat_kurt': [],
             'Fstat_median': [],
             'Fstat_min': [],
             'Fstat_P10': [],
             'Fstat_P90': [],
             'Fstat_max': [],
             'Fstat_iqr': [],
             'Fstat_range': [],
             'Fstat_mad': [],
             'Fstat_rmad': [],
             'Fstat_medad': [],
             'Fstat_cov': [],
             'Fstat_qcod': [],
             'Fstat_energy': [],
             'Fstat_rms': []}

    # STARTING COMPUTATION

    if definite:
        stats['Fstat_mean'] = np.mean(X)  # Mean
        stats['Fstat_var'] = np.var(X)  # Variance
        stats['Fstat_skew'] = skew(X)  # Skewness
        stats['Fstat_kurt'] = kurtosis(X)  # Kurtosis
        stats['Fstat_median'] = np.median(X)  # Median
        stats['Fstat_min'] = np.min(X)  # Minimum grey level
        stats['Fstat_P10'] = scoreatpercentile(X, 10)  # 10th percentile
        stats['Fstat_P90'] = scoreatpercentile(X, 90)  # 90th percentile
        stats['Fstat_max'] = np.max(X)  # Maximum grey level
        stats['Fstat_iqr'] = iqr(X)  # Interquantile range
        stats['Fstat_range'] = np.ptp(X)  # Range max(X) - min(X)

        # Mean absolute deviation
        stats['Fstat_mad'] = np.mean(np.absolute(X - np.mean(X)))

        X_10_90 = X[np.where((X >= stats['Fstat_P10']) &
                             (X <= stats['Fstat_P90']), True, False)]

        # Robust mean absolute deviation
        stats['Fstat_rmad'] = np.mean(np.abs(X_10_90 - np.mean(X_10_90)))

        # Median absolute deviation
        stats['Fstat_medad'] = np.mean(np.absolute(X - np.median(X)))
        stats['Fstat_cov'] = variation(X)  # Coefficient of variation

        X_75_25 = scoreatpercentile(X, 75) + scoreatpercentile(X, 25)
        # Quartile coefficient of dispersion
        stats['Fstat_qcod'] = iqr(X)/X_75_25
        stats['Fstat_energy'] = np.sum(np.power(X, 2))  # Energy
        stats['Fstat_rms'] = np.sqrt(np.mean(np.power(X, 2)))  # Root mean square

    return stats
