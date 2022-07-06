#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

from ..biomarkers.get_ngtdm_matrix import get_ngtdm_matrix


def extract_all(vol, distCorrection=None) -> Dict:
    """Compute ngtdm features.

    Args:

        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region
            of interest.
        distCorrection (Union[bool, str], optional): Set this variable to true in order to use
            discretization length difference corrections as used here:
            <https://doi.org/10.1088/0031-9155/60/14/5471>.
            Set this variable to false to replicate IBSI results.
            Or use string and specify the norm for distance weighting. Weighting is 
            only performed if this argument is "manhattan", "euclidean" or "chebyshev".
    
    Returns:
        Dict: Dict of Neighbourhood grey tone difference based features.
    """

    ngtdm_features = {'Fngt_coarseness': [],
             'Fngt_contrast': [],
             'Fngt_busyness': [],
             'Fngt_complexity': [],
             'Fngt_strength': []}

    # GET THE ngtdm MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)

    if distCorrection is None:
        ngtdm, count_valid = get_ngtdm_matrix(vol, levels)
    else:
        ngtdm, count_valid = get_ngtdm_matrix(vol, levels, distCorrection)

    nTot = np.sum(count_valid)
    # Now representing the probability of gray-level occurences
    count_valid = count_valid/nTot
    NL = np.size(ngtdm)
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
        for i in range(1, NL+1):
            for j in range(1, NL+1):
                val = val + count_valid[i-1] * count_valid[j-1] * ((i-j)**2)
        ngtdm_features['Fngt_contrast'] = val * np.sum(ngtdm) / (n_g*(n_g-1)*nTot)

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
                p_valid[i-1]-p_valid[j-1]) / (nTot*(
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
