#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np


def get_ngtdm_matrix(roi_only:np.ndarray,
                     levels:np.ndarray,
                     dist_correction: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """Computes ngtdm matrix.

    This function computes the Neighborhood Gray-Tone Difference Matrix
    (ngtdm) of the region of interest (ROI) of an input volume. The input
    volume is assumed to be isotropically resampled. The ngtdm is computed
    using 26-voxel connectivity. To account for discretization length
    differences, all averages around a center voxel are performed such that
    the neighbours at a distance of sqrt(3) voxels are given a weight of
    1/sqrt(3), and the neighbours at a distance of sqrt(2) voxels are given a
    weight of 1/sqrt(2).

    Note:
        This function is compatible with 2D analysis (language not adapted in the text)

    REFERENCE:
        [1] Amadasun, M., & King, R. (1989). Textural Features Corresponding to
        Textural Properties. IEEE Transactions on Systems Man and Cybernetics,
        19(5), 1264–1274.
    
    Args:
        roi_only (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are set to NaNs.
        levels (ndarray): Vector containing the quantized gray-levels in the tumor region
            (or reconstruction levels of quantization).
        dist_correction (bool, optional): Set this variable to true in order to use
            discretization length difference corrections as used 
            here: https://doi.org/10.1088/0031-9155/60/14/5471.
            Set this variable to false to replicate IBSI results.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - ngtdm: Neighborhood Gray-Tone Difference Matrix of 'roi_only'.
            - count_valid: Array of number of valid voxels used in the ngtdm computation.

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
