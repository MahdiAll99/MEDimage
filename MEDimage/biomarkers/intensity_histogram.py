#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from typing import Dict, Tuple

import numpy as np
from scipy.stats import scoreatpercentile, variation


def init_IH(vol: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Initialize Intensity Histogram Features.

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

    # CONSTRUCTION OF HISTOGRAM AND ASSOCIATED NUMBER OF GRAY-LEVELS

    # Always defined from 1 to the maximum value of
    # the volume to remove any ambiguity
    levels = np.arange(1, np.max(x) + 100*np.finfo(float).eps)
    n_g = levels.size  # Number of gray-levels
    h = np.zeros(n_g)  # The histogram of x

    for i in np.arange(0, n_g):
        # == i or == levels(i) is equivalent since levels = 1:max(x),
        # and n_g = numel(levels)
        h[i] = np.sum(x == i + 1)  # h[i] = sum(x == i+1)

    p = (h / n_v)  # Occurence probability for each grey level bin i
    pt = p.transpose()

    return x, levels, n_g, h, p, pt

def extract_all(vol: np.ndarray) -> Dict:
    """Computes Intensity Histogram Features.
    These features refer to Intensity histogram family in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, QUANTIZED (e.g. nBins = 100,
                       levels = [1, ..., max]), with NaNs outside the region of interest.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    """
    warnings.simplefilter("ignore")

    # INITIALIZATION
    x, levels, n_g, h, p, pt = init_IH(vol)

    # Initialization of final structure (Dictionary) containing all features.
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
               'Fih_min_grad_gl': []
               }

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
        skew = np.matmul(np.power(levels - u, 3), pt) / np.power(var, 3/2)
        kurt = np.matmul(np.power(levels - u, 4), pt) / np.power(var, 2) - 3

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

    for i in np.arange(1, n_g-1):
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

def mean(vol: np.ndarray) -> float:
    """Compute Intensity histogram mean feature of the input dataset (3D Array).
    This feature refers to "Fih_mean" (ID = X6K6) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram mean
    """
    _, levels, _, _, _, pt = init_IH(vol)  # Initialization
    
    return np.matmul(levels, pt) # Intensity histogram mean

def var(vol: np.ndarray) -> float:
    """Compute Intensity histogram variance feature of the input dataset (3D Array).
    This feature refers to "Fih_var" (ID = CH89) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram variance
    """
    _, levels, _, _, _, pt = init_IH(vol)  # Initialization
    u = np.matmul(levels, pt) # Intensity histogram mean

    return np.matmul(np.power(levels - u, 2), pt)  # Intensity histogram variance

def skewness(vol: np.ndarray) -> float:
    """Compute Intensity histogram skewness feature of the input dataset (3D Array).
    This feature refers to "Fih_skew" (ID = 88K1) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram skewness.
    """
    _, levels, _, _, _, pt = init_IH(vol)  # Initialization
    u = np.matmul(levels, pt) # Intensity histogram mean
    var = np.matmul(np.power(levels - u, 2), pt)  # Intensity histogram variance
    if var != 0:
        skew = np.matmul(np.power(levels - u, 3), pt) / np.power(var, 3/2)

    return skew  # Skewness

def kurt(vol: np.ndarray) -> float:
    """Compute Intensity histogram kurtosis feature of the input dataset (3D Array).
    This feature refers to "Fih_kurt" (ID = C3I7) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: The Intensity histogram kurtosis feature
    """
    _, levels, _, _, _, pt = init_IH(vol)  # Initialization
    u = np.matmul(levels, pt) # Intensity histogram mean
    var = np.matmul(np.power(levels - u, 2), pt)  # Intensity histogram variance
    if var != 0:
        kurt = np.matmul(np.power(levels - u, 4), pt) / np.power(var, 2) - 3

    return kurt  # Kurtosis

def median(vol: np.ndarray) -> float:
    """Compute Intensity histogram median feature along the specified axis of the input dataset (3D Array).
    This feature refers to "Fih_median" (ID = WIFQ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram median feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.median(x)  # Median

def min(vol: np.ndarray) -> float:
    """Compute Intensity histogram minimum grey level feature of the input dataset (3D Array).
    This feature refers to "Fih_min" (ID = 1PR8) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram minimum grey level feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.min(x)  # Minimum grey level

def p10(vol: np.ndarray) -> float:
    """Compute Intensity histogram 10th percentile feature of the input dataset (3D Array).
    This feature refers to "Fih_P10" (ID = GPMT) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram 10th percentile feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 10)  # 10th percentile

def p90(vol: np.ndarray) -> float:
    """Compute Intensity histogram 90th percentile feature of the input dataset (3D Array).
    This feature refers to "Fih_P90" (ID = OZ0C) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    Returns:
        float: Intensity histogram 90th percentile feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 90)  # 90th percentile

def max(vol: np.ndarray) -> float:
    """Compute Intensity histogram maximum grey level feature of the input dataset (3D Array).
    This feature refers to "Fih_max" (ID = 3NCY) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram maximum grey level feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.max(x)  # Maximum grey level

def mode(vol: np.ndarray) -> int:
    """Compute Intensity histogram mode feature of the input dataset (3D Array).
    This feature refers to "Fih_mode" (ID = AMMC) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        integer: Intensity histogram mode. 
        levels = 1:max(x), so the index of the ith bin of h is the same as i
    """
    _, levels, _, h, _, pt = init_IH(vol)  # Initialization
    u = np.matmul(levels, pt)
    mh = np.max(h)
    mode = np.where(h == mh)[0] + 1

    if np.size(mode) > 1:
        dist = np.abs(mode - u)
        ind_min = np.argmin(dist)

        return mode[ind_min]  # Intensity histogram mode.
    else:

        return mode[0]  # Intensity histogram mode.

def iqrange(vol: np.ndarray) -> float:
    r"""Compute Intensity histogram interquantile range feature of the input dataset (3D Array).
    This feature refers to "Fih_iqr" (ID = WR0O) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Interquartile range. If :math:`axis ≠ None` , the output data-type is the same as that of the input.
        Since x goes from :math:`1:max(x)` , all with integer values, the result is an integer.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 75) - scoreatpercentile(x, 25)  # Intensity histogram interquantile range

def range(vol: np.ndarray) -> float:
    """Compute Intensity histogram range of values (maximum - minimum) feature of the input dataset (3D Array).
    This feature refers to "Fih_range" (ID = 5Z3W) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram range.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.max(x) - np.min(x) # Intensity histogram range

def mad(vol: np.ndarray) -> float:
    """Compute Intensity histogram mean absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fih_mad" (ID = D2ZX) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float : Intensity histogram mean absolute deviation feature.
    """
    x, levels, _, _, _, pt = init_IH(vol)  # Initialization
    u = np.matmul(levels, pt)

    return np.mean(abs(x - u))  # Intensity histogram mean absolute deviation

def rmad(vol: np.ndarray) -> float:
    """Compute Intensity histogram robust mean absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fih_rmad" (ID = WRZB) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
        P10(ndarray): Score at 10th percentil.
        P90(ndarray): Score at 90th percentil.
    
    Returns:
        float: Intensity histogram robust mean absolute deviation
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    P10 = scoreatpercentile(x, 10)  # 10th percentile
    P90 = scoreatpercentile(x, 90)  # 90th percentile
    x_10_90 = x[np.where((x >= P10) &
                         (x <= P90), True, False)]  # Holding x for (x >= P10) and (x<= P90)

    return np.mean(np.abs(x_10_90 - np.mean(x_10_90)))  # Intensity histogram robust mean absolute deviation

def medad(vol: np.ndarray) -> float:
    """Intensity histogram median absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fih_medad" (ID = 4RNL) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram median absolute deviation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(np.absolute(x - np.median(x)))  # Intensity histogram median absolute deviation

def cov(vol: np.ndarray) -> float:
    """Compute Intensity histogram coefficient of variation feature of the input dataset (3D Array).
    This feature refers to "Fih_cov" (ID = CWYJ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram coefficient of variation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return variation(x)  # Intensity histogram coefficient of variation

def qcod(vol: np.ndarray) -> float:
    """Compute the quartile coefficient of dispersion feature of the input dataset (3D Array).
    This feature refers to "Fih_qcod" (ID = SLWD) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        ndarray: A new array holding the quartile coefficient of dispersion feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)     

    return iqrange(x) / x_75_25  # Quartile coefficient of dispersion

def entropy(vol: np.ndarray) -> float:
    """Compute Intensity histogram entropy feature of the input dataset (3D Array).
    This feature refers to "Fih_entropy" (ID = TLU2) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram entropy feature.
    """
    x, _, _, _, p, _ = init_IH(vol)  # Initialization
    p = p[p > 0]

    return -np.sum(p * np.log2(p))  # Intensity histogram entropy

def uniformity(vol: np.ndarray) -> float:
    """Compute Intensity histogram uniformity feature of the input dataset (3D Array).
    This feature refers to "Fih_uniformity" (ID = BJ5W) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Intensity histogram uniformity feature.
    """
    x, _, _, _, p, _ = init_IH(vol)  # Initialization
    p = p[p > 0]

    return np.sum(np.power(p, 2))  # Intensity histogram uniformity

def hist_grad_calc(vol: np.ndarray) -> np.ndarray:
    """Calculation of histogram gradient.
    This feature refers to "Fih_hist_grad_calc" (ID = 12CE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        ndarray: Histogram gradient
    """
    _, _, n_g, h, _, _ = init_IH(vol)  # Initialization
    hist_grad = np.zeros(n_g)
    hist_grad[0] = h[1] - h[0]
    hist_grad[-1] = h[-1] - h[-2]
    for i in np.arange(1, n_g-1):
        hist_grad[i] = (h[i+1] - h[i-1])/2

    return hist_grad  # Intensity histogram uniformity

def max_grad(vol: np.ndarray) -> float:
    """Calculation of Maximum histogram gradient feature.
    This feature refers to "Fih_max_grad" (ID = 12CE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Maximum histogram gradient feature.
    """
    hist_grad = hist_grad_calc(vol)  # Initialization

    return np.max(hist_grad)  # Maximum histogram gradient

def max_grad_gl(vol: np.ndarray) -> float:
    """Calculation of Maximum histogram gradient grey level feature.
    This feature refers to "Fih_max_grad_gl" (ID = 8E6O) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Maximum histogram gradient grey level feature.
    """
    _, levels, _, _, _, _ = init_IH(vol)  # Initialization
    hist_grad = hist_grad_calc(vol) 
    ind_max = np.where(hist_grad == np.max(hist_grad))[0][0]  

    return levels[ind_max]  # Maximum histogram gradient grey level

def min_grad(vol: np.ndarray) -> float:
    """Calculation of Minimum histogram gradient feature.
    This feature refers to "Fih_min_grad" (ID = VQB3) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Minimum histogram gradient feature.
    """
    hist_grad = hist_grad_calc(vol)  # Initialization

    return np.min(hist_grad)  # Minimum histogram gradient

def min_grad_gl(vol: np.ndarray) -> float:
    """Calculation of Minimum histogram gradient grey level feature.
    This feature refers to "Fih_min_grad_gl" (ID = RHQZ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
    
    Returns:
        float: Minimum histogram gradient grey level feature.
    """
    _, levels, _, _, _, _ = init_IH(vol)  # Initialization
    hist_grad = hist_grad_calc(vol) 
    ind_min = np.where(hist_grad == np.min(hist_grad))[0][0] 

    return levels[ind_min]  # Minimum histogram gradient grey level
