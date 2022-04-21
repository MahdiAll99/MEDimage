#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np
import skimage.measure as skim


def getGLSZMmatrix(ROIOnly, levels) -> Dict:
    """Compute GLSZMmatrix.

    This function computes the Gray-Level Size Zone Matrix (GLSZM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. The zones of different sizes are computed
    using 26-voxel connectivity.

    Note:
        This function is compatible with 2D analysis (language not adapted in the text).

    Args:
        ROIOnlyInt (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).

    Returns:
        ndarray: Array of Gray-Level Size Zone Matrix of 'ROIOnly'.

    REFERENCES:
        [1] Thibault, G., Fertil, B., Navarro, C., Pereira, S., Cau, P., Levy,
            N., Mari, J.-L. (2009). Texture Indexes and Gray Level Size Zone
            Matrix. Application to Cell Nuclei Classification. In Pattern
            Recognition and Information Processing (PRIP) (pp. 140–145).
    
    """

    # PRELIMINARY
    ROIOnly = ROIOnly.copy()
    nMax = np.sum(~np.isnan(ROIOnly))
    levelTemp = np.max(levels) + 1
    ROIOnly[np.isnan(ROIOnly)] = levelTemp
    levels = np.append(levels, levelTemp)

    # QUANTIZATION EFFECTS CORRECTION
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    uniqueVect = levels
    NL = np.size(levels) - 1

    # INITIALIZATION
    # THIS NEEDS TO BE CHANGED. THE ARRAY INITIALIZED COULD BE TOO BIG!
    GLSZM = np.zeros((NL, nMax))

    # COMPUTATION OF GLSZM
    temp = ROIOnly.copy().astype('int')
    for i in range(1, NL+1):
        temp[ROIOnly != uniqueVect[i-1]] = 0
        temp[ROIOnly == uniqueVect[i-1]] = 1
        connObjects, nZone = skim.label(temp, return_num=True)
        for j in range(1, nZone+1):
            col = np.sum(connObjects == j)
            GLSZM[i-1, col-1] = GLSZM[i-1, col-1] + 1

    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(GLSZM, 0))[0][-1]
    GLSZM = np.delete(GLSZM, range(stop+1, np.shape(GLSZM)[1]), 1)

    return GLSZM
