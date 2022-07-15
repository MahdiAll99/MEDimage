#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Union

import numpy as np
import skimage.measure as skim


def get_glszm_matrix(roi_only: np.ndarray, 
                     levels: Union[np.ndarray, List]) -> Dict:
    """Compute GLSZMmatrix.

    This function computes the Gray-Level Size Zone Matrix (GLSZM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. The zones of different sizes are computed
    using 26-voxel connectivity.

    Note:
        This function is compatible with 2D analysis (language not adapted in the text).

    Args:
        roi_only_int (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels
            in the tumor region (or reconstruction levels of quantization).

    Returns:
        ndarray: Array of Gray-Level Size Zone Matrix of 'roi_only'.

    REFERENCES:
        [1] Thibault, G., Fertil, B., Navarro, C., Pereira, S., Cau, P., Levy,
            N., Mari, J.-L. (2009). Texture Indexes and Gray Level Size Zone
            Matrix. Application to Cell Nuclei Classification. In Pattern
            Recognition and Information Processing (PRIP) (pp. 140–145).
    """

    # PRELIMINARY
    roi_only = roi_only.copy()
    n_max = np.sum(~np.isnan(roi_only))
    level_temp = np.max(levels) + 1
    roi_only[np.isnan(roi_only)] = level_temp
    levels = np.append(levels, level_temp)

    # QUANTIZATION EFFECTS CORRECTION
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    unique_vect = levels
    n_l = np.size(levels) - 1

    # INITIALIZATION
    # THIS NEEDS TO BE CHANGED. THE ARRAY INITIALIZED COULD BE TOO BIG!
    glszm = np.zeros((n_l, n_max))

    # COMPUTATION OF glszm
    temp = roi_only.copy().astype('int')
    for i in range(1, n_l+1):
        temp[roi_only != unique_vect[i-1]] = 0
        temp[roi_only == unique_vect[i-1]] = 1
        conn_objects, n_zone = skim.label(temp, return_num=True)
        for j in range(1, n_zone+1):
            col = np.sum(conn_objects == j)
            glszm[i-1, col-1] = glszm[i-1, col-1] + 1

    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(glszm, 0))[0][-1]
    glszm = np.delete(glszm, range(stop+1, np.shape(glszm)[1]), 1)

    return glszm
