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
    x = vol[~np.isnan(vol[:])]
    return np.mean(x)

def extract_all(vol, intensity=None):
    """Compute stats.

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

    x = vol[~np.isnan(vol[:])]

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
        stats['Fstat_mean'] = np.mean(x)  # Mean
        stats['Fstat_var'] = np.var(x)  # Variance
        stats['Fstat_skew'] = skew(x)  # Skewness
        stats['Fstat_kurt'] = kurtosis(x)  # Kurtosis
        stats['Fstat_median'] = np.median(x)  # Median
        stats['Fstat_min'] = np.min(x)  # Minimum grey level
        stats['Fstat_P10'] = scoreatpercentile(x, 10)  # 10th percentile
        stats['Fstat_P90'] = scoreatpercentile(x, 90)  # 90th percentile
        stats['Fstat_max'] = np.max(x)  # Maximum grey level
        stats['Fstat_iqr'] = iqr(x)  # Interquantile range
        stats['Fstat_range'] = np.ptp(x)  # Range max(x) - min(x)

        # Mean absolute deviation
        stats['Fstat_mad'] = np.mean(np.absolute(x - np.mean(x)))

        x_10_90 = x[np.where((x >= stats['Fstat_P10']) &
                             (x <= stats['Fstat_P90']), True, False)]

        # Robust mean absolute deviation
        stats['Fstat_rmad'] = np.mean(np.abs(x_10_90 - np.mean(x_10_90)))

        # Median absolute deviation
        stats['Fstat_medad'] = np.mean(np.absolute(x - np.median(x)))
        stats['Fstat_cov'] = variation(x)  # Coefficient of variation

        x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)
        # Quartile coefficient of dispersion
        stats['Fstat_qcod'] = iqr(x)/x_75_25
        stats['Fstat_energy'] = np.sum(np.power(x, 2))  # Energy
        stats['Fstat_rms'] = np.sqrt(np.mean(np.power(x, 2)))  # Root mean square

    return stats

