#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy

import numpy as np
from numpy import ndarray
from skimage.exposure import equalize_hist


def equalization(vol_RE) -> ndarray:
    """
    Performs histogram equalisation of the ROI imaging intensities.

    Note:
        THIS IS A PURE "WHAT IS CONTAINED WITHIN THE ROI" EQUALIZATION. THIS IS
        NOT INFLUENCED BY THE "userSetMinVal" USED FOR FBS DISCRESTISATION.

    Args:
        vol_RE (ndarray): 3D array of the image volume that will be studied with 
            NaN value for the excluded voxels (voxels outside the ROI mask).

    Returns:
        ndarray: Same input image volume but with redistributed intensities.

    """

    # AZ: This was made part of the function call
    # Ng = 64
    # This is the default we will use. It means that when using 'FBS',
    # nQ should be chosen wisely such
    # that the total number of grey levels does not exceed 64, for all
    # patients (recommended).
    # This choice was amde by considering that the best equalization
    # performance for "histeq.m" is obtained with low Ng.
    # WARNING: The effective number of grey levels coming out of "histeq.m"
    # may be lower than Ng.

    # CONSERVE THE INDICES OF THE ROI
    Xgl = np.ravel(vol_RE)
    indROI = np.where(~np.isnan(vol_RE))
    Xgl = Xgl[~np.isnan(Xgl)]

    # ADJUST RANGE BETWEEN 0 and 1
    minVal = np.min(Xgl)
    maxVal = np.max(Xgl)
    Xgl01 = (Xgl - minVal)/(maxVal - minVal)

    # EQUALIZATION
    # Xgl_equal = equalize_hist(Xgl01, nbins=Ng)
    # AT THE MOMENT, WE CHOOSE TO USE THE DEFAULT NUMBER OF BINS OF
    # equalize_hist.py (256)
    Xgl_equal = equalize_hist(Xgl01)
    # RE-ADJUST TO CORRECT RANGE
    Xgl_equal = (Xgl_equal - np.min(Xgl_equal)) / \
        (np.max(Xgl_equal) - np.min(Xgl_equal))
    Xgl_equal = Xgl_equal * (maxVal - minVal)
    Xgl_equal = Xgl_equal + minVal

    # RECONSTRUCT THE VOLUME WITH EQUALIZED VALUES
    volEqual_RE = deepcopy(vol_RE)

    volEqual_RE[indROI] = Xgl_equal

    return volEqual_RE
