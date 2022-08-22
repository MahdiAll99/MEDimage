#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from typing import Dict, Tuple, Union

import numpy as np


def get_matrix(roi_only:np.ndarray,
                     levels:np.ndarray,
                     dist_correction: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """This function computes the Neighborhood Gray-Tone Difference Matrix
    (NGTDM) of the region of interest (ROI) of an input volume. The input
    volume is assumed to be isotropically resampled. The ngtdm is computed
    using 26-voxel connectivity. To account for discretization length
    differences, all averages around a center voxel are performed such that
    the neighbours at a distance of :math:`\sqrt{3}` voxels are given a weight of
    :math:`\sqrt{3}`, and the neighbours at a distance of :math:`\sqrt{2}` voxels are given a
    weight of :math:`\sqrt{2}`.
    This matrix refers to "Neighbourhood grey tone difference based features" (ID = IPET)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Note:
        This function is compatible with 2D analysis (language not adapted in the text)

    Args:
        roi_only (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are set to NaNs.
        levels (ndarray): Vector containing the quantized gray-levels in the tumor region
            (or reconstruction ``levels`` of quantization).
        dist_correction (bool, optional): Set this variable to true in order to use
            discretization length difference corrections as used by the `Institute of Physics and
            Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`_.
            Set this variable to false to replicate IBSI results.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - ngtdm: Neighborhood Gray-Tone Difference Matrix of ``roi_only'``.
            - count_valid: Array of number of valid voxels used in the ngtdm computation.

    REFERENCES:
        [1] Amadasun, M., & King, R. (1989). Textural Features Corresponding to
        Textural Properties. IEEE Transactions on Systems Man and Cybernetics,
        19(5), 1264–1274.
    
    """

    # PARSING "dist_correction" ARGUMENT
    if type(dist_correction) is not bool:
        # The user did not input either "true" or "false",
        # so the default behavior is used.
        dist_correction = True  # By default

    # PRELIMINARY
    if np.size(np.shape(roi_only)) == 2:  # generalization to 2D inputs
        two_d = 1
    else:
        two_d = 0

    roi_only = np.pad(roi_only, [1, 1], 'constant', constant_values=np.NaN)

    # # QUANTIZATION EFFECTS CORRECTION
    # # In case (for example) we initially wanted to have 64 levels, but due to
    # # quantization, only 60 resulted.
    unique_vol = levels.astype('int')
    NL = np.size(levels)
    temp = roi_only.copy().astype('int')
    for i in range(1, NL+1):
        roi_only[temp == unique_vol[i-1]] = i

    # INTIALIZATION
    ngtdm = np.zeros(NL)
    count_valid = np.zeros(NL)

    # COMPUTATION OF ngtdm
    if two_d:
        indices = np.where(~np.isnan(np.reshape(
            roi_only, np.size(roi_only), order='F')))[0]
        pos_valid = np.unravel_index(indices, np.shape(roi_only), order='F')
        n_valid_temp = np.size(pos_valid[0])
        w4 = 1
        if dist_correction:
            # Weights given to different neighbors to correct
            # for discretization length differences
            w8 = 1/np.sqrt(2)
        else:
            w8 = 1

        weights = np.array([w8, w4, w8, w4, w4, w4, w8, w4, w8])

        for n in range(1, n_valid_temp+1):

            neighbours = roi_only[(pos_valid[0][n-1]-1):(pos_valid[0][n-1]+2),
                                 (pos_valid[1][n-1]-1):(pos_valid[1][n-1]+2)].copy()
            neighbours = np.reshape(neighbours, 9, order='F')
            neighbours = neighbours*weights
            value = neighbours[4].astype('int')
            neighbours[4] = np.NaN
            neighbours = neighbours/np.sum(weights[~np.isnan(neighbours)])
            neighbours = np.delete(neighbours, 4)  # Remove the center voxel
            # Thus only excluding voxels with NaNs only as neighbors.
            if np.size(neighbours[~np.isnan(neighbours)]) > 0:
                ngtdm[value-1] = ngtdm[value-1] + np.abs(
                    value-np.sum(neighbours[~np.isnan(neighbours)]))
                count_valid[value-1] = count_valid[value-1] + 1
    else:

        indices = np.where(~np.isnan(np.reshape(
            roi_only, np.size(roi_only), order='F')))[0]
        pos_valid = np.unravel_index(indices, np.shape(roi_only), order='F')
        n_valid_temp = np.size(pos_valid[0])
        w6 = 1
        if dist_correction:
            # Weights given to different neighbors to correct
            # for discretization length differences
            w26 = 1 / np.sqrt(3)
            w18 = 1 / np.sqrt(2)
        else:
            w26 = 1
            w18 = 1

        weights = np.array([w26, w18, w26, w18, w6, w18, w26, w18, w26, w18,
                            w6, w18, w6, w6, w6, w18, w6, w18, w26, w18,
                            w26, w18, w6, w18, w26, w18, w26])

        for n in range(1, n_valid_temp+1):
            neighbours = roi_only[(pos_valid[0][n-1]-1) : (pos_valid[0][n-1]+2),
                                 (pos_valid[1][n-1]-1) : (pos_valid[1][n-1]+2),
                                 (pos_valid[2][n-1]-1) : (pos_valid[2][n-1]+2)].copy()
            neighbours = np.reshape(neighbours, 27, order='F')
            neighbours = neighbours * weights
            value = neighbours[13].astype('int')
            neighbours[13] = np.NaN
            neighbours = neighbours / np.sum(weights[~np.isnan(neighbours)])
            neighbours = np.delete(neighbours, 13)  # Remove the center voxel
            # Thus only excluding voxels with NaNs only as neighbors.
            if np.size(neighbours[~np.isnan(neighbours)]) > 0:
                ngtdm[value-1] = ngtdm[value-1] + np.abs(value - np.sum(neighbours[~np.isnan(neighbours)]))
                count_valid[value-1] = count_valid[value-1] + 1

    return ngtdm, count_valid

def extract_all(vol: np.ndarray,
                dist_correction :Union[bool, str]=None) -> Dict:
    """Compute Neighbourhood grey tone difference based features.
    These features refer to "Neighbourhood grey tone difference based features" (ID = IPET) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:

        vol (ndarray): 3D volume, isotropically resampled, quantized
                       (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region
                       of interest.
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used
                                                      by the `Institute of Physics and Engineering in
                                                      Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
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
    ngtdm, count_valid = get_matrix(vol, levels, dist_correction)

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

def get_single_matrix(vol: np.ndarray,
               dist_correction = None)-> Tuple[np.ndarray,
                                               np.ndarray]:
    """Compute neighbourhood grey tone difference matrix in order to compute the single features.

    Args:

        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region of interest.
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used
                                                      by the `Institute of Physics and Engineering in
                                                      Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
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

    ngtdm, count_valid = get_matrix(vol, levels, dist_correction)
    
    return ngtdm, count_valid

def coarseness(ngtdm: np.ndarray, count_valid: np.ndarray)-> float:
    """
    Computes coarseness feature.
    This feature refers to "Coarseness" (ID = QCDE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
