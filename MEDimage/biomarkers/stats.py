#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import iqr, kurtosis, skew, scoreatpercentile, variation


def extract_all(vol: np.ndarray, intensity_type: str) -> dict:
    """Computes Intensity-based statistical features.
    These features refer to "Intensity-based statistical features" (ID = UHIW) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution).
        intensity_type (str): Type of intensity to compute. Can be "arbitrary", "definite" or "filtered".
            Will compute features only for "definite" intensity type.

    Return:
        dict: Dictionnary containing all stats features.

    Raises:
        ValueError: If `intensity_type` is not "arbitrary", "definite" or "filtered".
    """
    assert intensity_type in ["arbitrary", "definite", "filtered"], \
        "intensity_type must be 'arbitrary', 'definite' or 'filtered'"
    
    x = vol[~np.isnan(vol[:])]  # Initialization

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
             'Fstat_rms': []
             }

    # STARTING COMPUTATION
    if intensity_type == "definite":
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
        stats['Fstat_mad'] = np.mean(np.absolute(x - np.mean(x)))  # Mean absolute deviation
        x_10_90 = x[np.where((x >= stats['Fstat_P10']) &
                                (x <= stats['Fstat_P90']), True, False)]
        stats['Fstat_rmad'] = np.mean(np.abs(x_10_90 - np.mean(x_10_90)))  # Robust mean absolute deviation
        stats['Fstat_medad'] = np.mean(np.absolute(x - np.median(x)))  # Median absolute deviation
        stats['Fstat_cov'] = variation(x)  # Coefficient of variation
        x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)
        stats['Fstat_qcod'] = iqr(x)/x_75_25  # Quartile coefficient of dispersion
        stats['Fstat_energy'] = np.sum(np.power(x, 2))  # Energy
        stats['Fstat_rms'] = np.sqrt(np.mean(np.power(x, 2)))  # Root mean square

    return stats

def mean(vol: np.ndarray) -> float:
    """Computes statistical mean feature of the input dataset (3D Array).
    This feature refers to "Fstat_mean" (ID = Q4LE)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: Statistical mean feature
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(x)  # Mean

def var(vol: np.ndarray) -> float:
    """Computes statistical variance feature of the input dataset (3D Array).
    This feature refers to "Fstat_var" (ID = ECT3)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: Statistical variance feature
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.var(x)  # Variance

def skewness(vol: np.ndarray) -> float:
    """Computes the sample skewness feature of the input dataset (3D Array).
    This feature refers to "Fstat_skew" (ID = KE2A)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: The skewness feature of values along an axis. Returning 0 where all values are
        equal.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return skew(x)  # Skewness

def kurt(vol: np.ndarray) -> float:
    """Computes the kurtosis (Fisher or Pearson) feature of the input dataset (3D Array).
    This feature refers to "Fstat_kurt" (ID = IPH6)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: The kurtosis feature of values along an axis. If all values are equal,
        return -3 for Fisher's definition and 0 for Pearson's definition.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return kurtosis(x)  # Kurtosis

def median(vol: np.ndarray) -> float:
    """Computes the median feature along the specified axis of the input dataset (3D Array).
    This feature refers to "Fstat_median" (ID = Y12H)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: The median feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.median(x)  # Median

def min(vol: np.ndarray) -> float:
    """Computes the minimum grey level feature of the input dataset (3D Array).
    This feature refers to "Fstat_min" (ID = 1GSF)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: The minimum grey level feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.min(x)  # Minimum grey level

def p10(vol: np.ndarray) -> float:
    """Computes the score at the 10th percentile feature of the input dataset (3D Array).
    This feature refers to "Fstat_P10" (ID = QG58)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: Score at 10th percentil.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 10)  # 10th percentile

def p90(vol: np.ndarray) -> float:
    """Computes the score at the 90th percentile feature of the input dataset (3D Array).
    This feature refers to "Fstat_P90" (ID = 8DWT)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: Score at 90th percentil.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return scoreatpercentile(x, 90)  # 90th percentile

def max(vol: np.ndarray) -> float:
    """Computes the maximum grey level feature of the input dataset (3D Array).
    This feature refers to "Fstat_max" (ID = 84IY)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: The maximum grey level feature of the array elements.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.max(x)  # Maximum grey level

def iqrange(vol: np.ndarray) -> float:
    """Computes the interquartile range feature of the input dataset (3D Array).
    This feature refers to "Fstat_iqr" (ID = SALO)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: Interquartile range. If axis != None, the output data-type is the same as that of the input.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return iqr(x)  # Interquartile range

def range(vol: np.ndarray) -> float:
    """Range of values (maximum - minimum) feature along an axis of the input dataset (3D Array).
    This feature refers to "Fstat_range" (ID = 2OJQ)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the range of values, unless out was specified,
        in which case a reference to out is returned.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.ptp(x)  # Range max(x) - min(x) 

def mad(vol: np.ndarray) -> float:
    """Mean absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fstat_mad" (ID = 4FUA)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float : A new array holding mean absolute deviation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(np.absolute(x - np.mean(x)))  # Mean absolute deviation

def rmad(vol: np.ndarray) -> float:
    """Robust mean absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fstat_rmad" (ID = 1128)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)
        P10(ndarray): Score at 10th percentil.
        P90(ndarray): Score at 90th percentil.

    Returns:
        float: A new array holding the robust mean absolute deviation.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    P10 = scoreatpercentile(x, 10)  # 10th percentile
    P90 = scoreatpercentile(x, 90)  # 90th percentile
    x_10_90 = x[np.where((x >= P10) &
                         (x <= P90), True, False)]  # Holding x for (x >= P10) and (x<= P90)
                         
    return np.mean(np.abs(x_10_90 - np.mean(x_10_90)))  # Robust mean absolute deviation

def medad(vol: np.ndarray) -> float:
    """Median absolute deviation feature of the input dataset (3D Array).
    This feature refers to "Fstat_medad" (ID = N72L)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the median absolute deviation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.mean(np.absolute(x - np.median(x)))  # Median absolute deviation

def cov(vol: np.ndarray) -> float:
    """Computes the coefficient of variation feature of the input dataset (3D Array).
    This feature refers to "Fstat_cov" (ID = 7TET)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the coefficient of variation feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return variation(x)  # Coefficient of variation

def qcod(vol: np.ndarray) -> float:
    """Computes the quartile coefficient of dispersion feature of the input dataset (3D Array).
    This feature refers to "Fstat_qcod" (ID = 9S40)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the quartile coefficient of dispersion feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization
    x_75_25 = scoreatpercentile(x, 75) + scoreatpercentile(x, 25)  

    return iqr(x) / x_75_25  # Quartile coefficient of dispersion

def energy(vol: np.ndarray) -> float:
    """Computes the energy feature of the input dataset (3D Array).
    This feature refers to "Fstat_energy" (ID = N8CA)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the energy feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.sum(np.power(x, 2))  # Energy

def rms(vol: np.ndarray) -> float:
    """Computes the root mean square feature of the input dataset (3D Array).
    This feature refers to "Fstat_rms" (ID = 5ZWQ)  in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol(ndarray): 3D volume, NON-QUANTIZED, with NaNs outside the region of interest
            (continuous imaging intensity distribution)

    Returns:
        float: A new array holding the root mean square feature.
    """
    x = vol[~np.isnan(vol[:])]  # Initialization

    return np.sqrt(np.mean(np.power(x, 2)))  # Root mean square
