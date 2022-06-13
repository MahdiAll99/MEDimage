#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from typing import Dict

import numpy as np
from scipy.stats import scoreatpercentile, variation

def get_mean(vol):
    # INITIALIZATION
    X = vol[~np.isnan(vol[:])]
    n_v = X.size

    # Always defined from 1 to the maximum value of
    # the volume to remove any ambiguity
    levels = np.arange(1, np.max(X) + 100*np.finfo(float).eps)
    n_g = levels.size  # Number of gray-levels
    H = np.zeros(n_g)  # The histogram of X

    for i in range(0, n_g):
        H[i] = np.sum(X == i+1)

    p = (H/n_v)  # Occurrence probability for each grey level bin i
    pt = p.transpose()

    # Intensity histogram mean
    return np.matmul(levels, pt)

def extract_all(vol) -> Dict:
    """Computes Intensity Histogram Features.

    Args:
        vol (ndarray): 3D volume, QUANTIZED (e.g. nBins = 100, 
            levels = [1, ..., max]), with NaNs outside the region of interest.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    
    """
    warnings.simplefilter("ignore")

    # INITIALIZATION

    X = vol[~np.isnan(vol[:])]
    n_v = X.size

    int_hist = {'Fih_mean': [],
               'Fih_var': [],
               'Fih_skew': [],
               'Fih_kurt': [],
               'Fih_median': [],
               'Fih_min': [],
               'Fih_P10': [],
               'Fih_P90': [],
               'Fih_max': [],
               'Fih_mode': [],
               'Fih_iqr': [],
               'Fih_range': [],
               'Fih_mad': [],
               'Fih_rmad': [],
               'Fih_medad': [],
               'Fih_cov': [],
               'Fih_qcod': [],
               'Fih_entropy': [],
               'Fih_uniformity': [],
               'Fih_max_grad': [],
               'Fih_max_grad_gl': [],
               'Fih_min_grad': [],
               'Fih_min_grad_gl': []}

    # CONSTRUCTION OF HISTOGRAM AND ASSOCIATED NUMBER OF GRAY-LEVELS

    # Always defined from 1 to the maximum value of
    # the volume to remove any ambiguity
    levels = np.arange(1, np.max(X)+100*np.finfo(float).eps)
    n_g = levels.size  # Number of gray-levels
    H = np.zeros(n_g)  # The histogram of X

    for i in range(0, n_g):
        # == i or == levels(i) is equivalent since levels = 1:max(X),
        # and n_g = numel(levels)
        H[i] = np.sum(X == i+1)  # H[i] = sum(X == i+1)

    p = (H/n_v)  # Occurence probability for each grey level bin i
    pt = p.transpose()

    # STARTING COMPUTATION

    # Intensity histogram mean
    u = np.matmul(levels, pt)
    int_hist['Fih_mean'] = u

    # Intensity histogram variance
    var = np.matmul(np.power(levels - u, 2), pt)
    int_hist['Fih_var'] = var

    # Intensity histogram skewness and kurtosis
    skew = 0
    kurt = 0

    if var != 0:
        skew = np.matmul(np.power(levels - u, 3), pt)/np.power(var, 3/2)
        kurt = np.matmul(np.power(levels - u, 4), pt)/np.power(var, 2) - 3

    int_hist['Fih_skew'] = skew
    int_hist['Fih_kurt'] = kurt

    # Intensity histogram median
    int_hist['Fih_median'] = np.median(X)

    # Intensity histogram minimum grey level
    int_hist['Fih_min'] = np.min(X)

    # Intensity histogram 10th percentile
    p10 = scoreatpercentile(X, 10)
    int_hist['Fih_P10'] = p10

    # Intensity histogram 90th percentile
    p90 = scoreatpercentile(X, 90)
    int_hist['Fih_P90'] = p90

    # Intensity histogram maximum grey level
    int_hist['Fih_max'] = np.max(X)

    # Intensity histogram mode
    #    levels = 1:max(X), so the index of the ith bin of H is the same as i
    mh = np.max(H)
    mode = np.where(H == mh)[0] + 1

    if np.size(mode) > 1:
        dist = np.abs(mode - u)
        ind_min = np.argmin(dist)
        int_hist['Fih_mode'] = mode[ind_min]
    else:
        int_hist['Fih_mode'] = mode[0]

    # Intensity histogram interquantile range
    #    Since X goes from 1:max(X), all with integer values,
    #    the result is an integer
    int_hist['Fih_iqr'] = scoreatpercentile(X, 75) - scoreatpercentile(X, 25)

    # Intensity histogram range
    int_hist['Fih_range'] = np.max(X) - np.min(X)

    # Intensity histogram mean absolute deviation
    int_hist['Fih_mad'] = np.mean(abs(X - u))

    # Intensity histogram robust mean absolute deviation
    X_10_90 = X[np.where((X >= p10) & (X <= p90), True, False)]
    int_hist['Fih_rmad'] = np.mean(np.abs(X_10_90 - np.mean(X_10_90)))

    # Intensity histogram median absolute deviation
    int_hist['Fih_medad'] = np.mean(np.absolute(X - np.median(X)))

    # Intensity histogram coefficient of variation
    int_hist['Fih_cov'] = variation(X)

    # Intensity histogram quartile coefficient of dispersion
    X_75_25 = scoreatpercentile(X, 75) + scoreatpercentile(X, 25)
    int_hist['Fih_qcod'] = int_hist['Fih_iqr'] / X_75_25

    # Intensity histogram entropy
    p = p[p > 0]
    int_hist['Fih_entropy'] = -np.sum(p * np.log2(p))

    # Intensity histogram uniformity
    int_hist['Fih_uniformity'] = np.sum(np.power(p, 2))

    # Calculation of histogram gradient
    hist_grad = np.zeros(n_g)
    hist_grad[0] = H[1] - H[0]
    hist_grad[-1] = H[-1] - H[-2]

    for i in range(1, n_g-1):
        hist_grad[i] = (H[i+1] - H[i-1])/2

    # Maximum histogram gradient
    int_hist['Fih_max_grad'] = np.max(hist_grad)

    # Maximum histogram gradient grey level
    ind_max = np.where(hist_grad == int_hist['Fih_max_grad'])[0][0]
    int_hist['Fih_max_grad_gl'] = levels[ind_max]

    # Minimum histogram gradient
    int_hist['Fih_min_grad'] = np.min(hist_grad)

    # Minimum histogram gradient grey level
    ind_min = np.where(hist_grad == int_hist['Fih_min_grad'])[0][0]
    int_hist['Fih_min_grad_gl'] = levels[ind_min]

    return int_hist
