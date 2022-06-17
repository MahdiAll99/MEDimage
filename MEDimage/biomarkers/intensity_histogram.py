#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from typing import Dict

import numpy as np
from scipy.stats import scoreatpercentile, variation

def get_mean(vol):
    # INITIALIZATION
    x = vol[~np.isnan(vol[:])]
    n_v = x.size

    # Always defined from 1 to the maximum value of
    # the volume to remove any ambiguity
    levels = np.arange(1, np.max(x) + 100*np.finfo(float).eps)
    n_g = levels.size  # Number of gray-levels
    h = np.zeros(n_g)  # The histogram of x

    for i in range(0, n_g):
        h[i] = np.sum(x == i+1)

    p = (h/n_v)  # Occurrence probability for each grey level bin i
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

    x = vol[~np.isnan(vol[:])]
    n_v = x.size

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
    levels = np.arange(1, np.max(x)+100*np.finfo(float).eps)
    n_g = levels.size  # Number of gray-levels
    h = np.zeros(n_g)  # The histogram of x

    for i in range(0, n_g):
        # == i or == levels(i) is equivalent since levels = 1:max(x),
        # and n_g = numel(levels)
        h[i] = np.sum(x == i+1)  # h[i] = sum(x == i+1)

    p = (h/n_v)  # Occurence probability for each grey level bin i
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
    int_hist['Fih_median'] = np.median(x)

    # Intensity histogram minimum grey level
    int_hist['Fih_min'] = np.min(x)

    # Intensity histogram 10th percentile
    p10 = scoreatpercentile(x, 10)
    int_hist['Fih_P10'] = p10

    # Intensity histogram 90th percentile
    p90 = scoreatpercentile(x, 90)
    int_hist['Fih_P90'] = p90

    # Intensity histogram maximum grey level
    int_hist['Fih_max'] = np.max(x)

    # Intensity histogram mode
    #    levels = 1:max(x), so the index of the ith bin of h is the same as i
    mh = np.max(h)
    mode = np.where(h == mh)[0] + 1

    if np.size(mode) > 1:
        dist = np.abs(mode - u)
        ind_min = np.argmin(dist)
        int_hist['Fih_mode'] = mode[ind_min]
    else:
        int_hist['Fih_mode'] = mode[0]

    # Intensity histogram interquantile range
    #    Since x goes from 1:max(x), all with integer values,
    #    the result is an integer
    int_hist['Fih_iqr'] = scoreatpercentile(x, 75) - scoreatpercentile(x, 25)

    # Intensity histogram range
    int_hist['Fih_range'] = np.max(x) - np.min(x)

    # Intensity histogram mean absolute deviation
    int_hist['Fih_mad'] = np.mean(abs(x - u))

    # Intensity histogram robust mean absolute deviation
    x_10_90 = x[np.where((x >= p10) & (x <= p90), True, False)]
    int_hist['Fih_rmad'] = np.mean(np.abs(x_10_90 - np.mean(x_10_90)))

    # Intensity histogram median absolute deviation
    int_hist['Fih_medad'] = np.mean(np.absolute(x - np.median(x)))

    # Intensity histogram coefficient of variation
    int_hist['Fih_cov'] = variation(x)

    # Intensity histogram quartile coefficient of dispersion
    x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)
    int_hist['Fih_qcod'] = int_hist['Fih_iqr'] / x_75_25

    # Intensity histogram entropy
    p = p[p > 0]
    int_hist['Fih_entropy'] = -np.sum(p * np.log2(p))

    # Intensity histogram uniformity
    int_hist['Fih_uniformity'] = np.sum(np.power(p, 2))

    # Calculation of histogram gradient
    hist_grad = np.zeros(n_g)
    hist_grad[0] = h[1] - h[0]
    hist_grad[-1] = h[-1] - h[-2]

    for i in range(1, n_g-1):
        hist_grad[i] = (h[i+1] - h[i-1])/2

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
