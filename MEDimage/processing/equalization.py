#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy

import numpy as np
from skimage.exposure import equalize_hist


def equalization(vol_re) -> np.ndarray:
    """
    Performs histogram equalisation of the ROI imaging intensities.

    Note:
        THIS IS A PURE "WHAT IS CONTAINED WITHIN THE ROI" EQUALIZATION. THIS IS
        NOT INFLUENCED BY THE "user_set_min_val" USED FOR FBS DISCRESTISATION.

    Args:
        vol_re (ndarray): 3D array of the image volume that will be studied with 
            NaN value for the excluded voxels (voxels outside the ROI mask).

    Returns:
        ndarray: Same input image volume but with redistributed intensities.

    """

    # AZ: This was made part of the function call
    # n_g = 64
    # This is the default we will use. It means that when using 'FBS',
    # nq should be chosen wisely such
    # that the total number of grey levels does not exceed 64, for all
    # patients (recommended).
    # This choice was amde by considering that the best equalization
    # performance for "histeq.m" is obtained with low n_g.
    # WARNING: The effective number of grey levels coming out of "histeq.m"
    # may be lower than n_g.

    # CONSERVE THE INDICES OF THE ROI
    x_gl = np.ravel(vol_re)
    ind_roi = np.where(~np.isnan(vol_re))
    x_gl = x_gl[~np.isnan(x_gl)]

    # ADJUST RANGE BETWEEN 0 and 1
    min_val = np.min(x_gl)
    max_val = np.max(x_gl)
    x_gl_01 = (x_gl - min_val)/(max_val - min_val)

    # EQUALIZATION
    # x_gl_equal = equalize_hist(x_gl_01, nbins=n_g)
    # AT THE MOMENT, WE CHOOSE TO USE THE DEFAULT NUMBER OF BINS OF
    # equalize_hist.py (256)
    x_gl_equal = equalize_hist(x_gl_01)
    # RE-ADJUST TO CORRECT RANGE
    x_gl_equal = (x_gl_equal - np.min(x_gl_equal)) / \
        (np.max(x_gl_equal) - np.min(x_gl_equal))
    x_gl_equal = x_gl_equal * (max_val - min_val)
    x_gl_equal = x_gl_equal + min_val

    # RECONSTRUCT THE VOLUME WITH EQUALIZED VALUES
    vol_equal_re = deepcopy(vol_re)

    vol_equal_re[ind_roi] = x_gl_equal

    return vol_equal_re
