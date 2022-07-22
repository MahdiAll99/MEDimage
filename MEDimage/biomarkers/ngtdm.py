#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from typing import Dict, Tuple, Union

import numpy as np

from ..biomarkers.get_ngtdm_matrix import get_ngtdm_matrix


def extract_all(vol: np.ndarray,
                dist_correction :Union[bool, str]=None) -> Dict:
    """Compute Neighbourhood grey tone difference based features.
    These features refer to "Neighbourhood grey tone difference based features" (ID = IPET) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:

        vol (ndarray): 3D volume, isotropically resampled, quantized
                       (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region
                       of interest.
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used
                                                      by the `Institute of Physics and Engineering in
                                                      Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`_.
                                                      Set this variable to false to replicate IBSI results.
                                                      Or use string and specify the norm for distance weighting.
                                                      Weighting is only performed if this argument is
                                                      "manhattan", "euclidean" or "chebyshev".
    
    Returns:
        Dict: Dict of Neighbourhood grey tone difference based features.
    """
    ngtdm_features = {'Fngt_coarseness': [],
             'Fngt_contrast': [],
             'Fngt_busyness': [],
             'Fngt_complexity': [],
             'Fngt_strength': []}

    # GET THE NGTDM MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)
    ngtdm, count_valid = get_ngtdm_matrix(vol, levels, dist_correction)

    n_tot = np.sum(count_valid)
    # Now representing the probability of gray-level occurences
    count_valid = count_valid/n_tot
    nl = np.size(ngtdm)
    n_g = np.sum(count_valid != 0)
    p_valid = np.where(np.reshape(count_valid, np.size(
        count_valid), order='F') > 0)[0]+1
    n_valid = np.size(p_valid)

    # COMPUTING TEXTURES
    # Coarseness
    coarseness = 1 / np.matmul(np.transpose(count_valid), ngtdm)
    coarseness = min(coarseness, 10**6)
    ngtdm_features['Fngt_coarseness'] = coarseness

    # Contrast
    if n_g == 1:
        ngtdm_features['Fngt_contrast'] = 0
    else:
        val = 0
        for i in range(1, nl+1):
            for j in range(1, nl+1):
                val = val + count_valid[i-1] * count_valid[j-1] * ((i-j)**2)
        ngtdm_features['Fngt_contrast'] = val * np.sum(ngtdm) / (n_g*(n_g-1)*n_tot)

    # Busyness
    if n_g == 1:
        ngtdm_features['Fngt_busyness'] = 0
    else:
        denom = 0
        for i in range(1, n_valid+1):
            for j in range(1, n_valid+1):
                denom = denom + np.abs(p_valid[i-1]*count_valid[p_valid[i-1]-1] -
                                       p_valid[j-1]*count_valid[p_valid[j-1]-1])
        ngtdm_features['Fngt_busyness'] = np.matmul(np.transpose(count_valid), ngtdm) / denom

    # Complexity
    val = 0
    for i in range(1, n_valid+1):
        for j in range(1, n_valid+1):
            val = val + (np.abs(
                p_valid[i-1]-p_valid[j-1]) / (n_tot*(
                count_valid[p_valid[i-1]-1] +
                count_valid[p_valid[j-1]-1])))*(
                count_valid[p_valid[i-1]-1]*ngtdm[p_valid[i-1]-1] +
                count_valid[p_valid[j-1]-1]*ngtdm[p_valid[j-1]-1])

    ngtdm_features['Fngt_complexity'] = val

    # Strength
    if np.sum(ngtdm) == 0:
        ngtdm_features['Fngt_strength'] = 0
    else:
        val = 0
        for i in range(1, n_valid+1):
            for j in range(1, n_valid+1):
                val = val + (count_valid[p_valid[i-1]-1] + count_valid[p_valid[j-1]-1])*(
                    p_valid[i-1]-p_valid[j-1])**2

        ngtdm_features['Fngt_strength'] = val/np.sum(ngtdm)

    return ngtdm_features

def get_matrix(vol: np.ndarray,
               dist_correction = None)-> Tuple[np.ndarray,
                                               np.ndarray]:
    """Compute neighbourhood grey tone difference matrix.

    Args:

        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region of interest.
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used
                                                      by the `Institute of Physics and Engineering in
                                                      Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`_.
                                                      Set this variable to false to replicate IBSI results.
                                                      Or use string and specify the norm for distance weighting.
                                                      Weighting is only performed if this argument is
                                                      "manhattan", "euclidean" or "chebyshev".
    
    Returns:
        np.ndarray: array of neighbourhood grey tone difference matrix
    """
    # GET THE NGTDM MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)

    ngtdm, count_valid = get_ngtdm_matrix(vol, levels, dist_correction)
    
    return ngtdm, count_valid

def coarseness(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes coarseness feature.
    This feature refers to "Coarseness" (ID = QCDE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        ngtdm (ndarray): array of neighbourhood grey tone difference matrix
    
    Returns:
        float: coarseness value

    """
    n_tot = np.sum(count_valid)
    count_valid = count_valid/n_tot
    coarseness = 1 / np.matmul(np.transpose(count_valid), ngtdm)
    coarseness = min(coarseness, 10**6)
    return coarseness

def contrast(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes contrast feature.
    This feature refers to "Contrast" (ID = 65HE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        ngtdm (ndarray): array of neighbourhood grey tone difference matrix
    
    Returns:
        float: contrast value

    """
    n_tot = np.sum(count_valid)
    count_valid = count_valid/n_tot
    nl = np.size(ngtdm)
    n_g = np.sum(count_valid != 0)

    if n_g == 1:
        return 0
    else:
        val = 0
        for i in range(1, nl+1):
            for j in range(1, nl+1):
                val = val + count_valid[i-1] * count_valid[j-1] * ((i-j)**2)
        contrast = val * np.sum(ngtdm) / (n_g*(n_g-1)*n_tot)
        return contrast

def busyness(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes busyness feature.
    This feature refers to "Busyness" (ID = NQ30) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        ngtdm (ndarray): array of neighbourhood grey tone difference matrix
    
    Returns:
        float: busyness value

    """
    n_tot = np.sum(count_valid)
    count_valid = count_valid/n_tot
    n_g = np.sum(count_valid != 0)
    p_valid = np.where(np.reshape(count_valid, np.size(
        count_valid), order='F') > 0)[0]+1
    n_valid = np.size(p_valid)

    if n_g == 1:
        busyness = 0
        return busyness
    else:
        denom = 0
        for i in range(1, n_valid+1):
            for j in range(1, n_valid+1):
                denom = denom + np.abs(p_valid[i-1]*count_valid[p_valid[i-1]-1] -
                                       p_valid[j-1]*count_valid[p_valid[j-1]-1])
        busyness = np.matmul(np.transpose(count_valid), ngtdm) / denom 
        return busyness

def complexity(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes complexity feature.
    This feature refers to "Complexity" (ID = HDEZ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        ngtdm (ndarray): array of neighbourhood grey tone difference matrix
    
    Returns:
        float: complexity value

    """
    n_tot = np.sum(count_valid)
    # Now representing the probability of gray-level occurences
    count_valid = count_valid/n_tot
    p_valid = np.where(np.reshape(count_valid, np.size(
        count_valid), order='F') > 0)[0]+1
    n_valid = np.size(p_valid)

    val = 0
    for i in range(1, n_valid+1):
        for j in range(1, n_valid+1):
            val = val + (np.abs(
                p_valid[i-1]-p_valid[j-1]) / (n_tot*(
                count_valid[p_valid[i-1]-1] +
                count_valid[p_valid[j-1]-1])))*(
                count_valid[p_valid[i-1]-1]*ngtdm[p_valid[i-1]-1] +
                count_valid[p_valid[j-1]-1]*ngtdm[p_valid[j-1]-1])
    complexity = val
    return complexity

def strength(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes strength feature.
    This feature refers to "Strength" (ID = 1X9X) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        ngtdm (ndarray): array of neighbourhood grey tone difference matrix
    
    Returns:
        float: strength value

    """

    n_tot = np.sum(count_valid)
    # Now representing the probability of gray-level occurences
    count_valid = count_valid/n_tot
    p_valid = np.where(np.reshape(count_valid, np.size(
        count_valid), order='F') > 0)[0]+1
    n_valid = np.size(p_valid)

    if np.sum(ngtdm) == 0:
        strength = 0
        return strength
    else:
        val = 0
        for i in range(1, n_valid+1):
            for j in range(1, n_valid+1):
                val = val + (count_valid[p_valid[i-1]-1] + count_valid[p_valid[j-1]-1])*(
                    p_valid[i-1]-p_valid[j-1])**2

        strength = val/np.sum(ngtdm)
        return strength
