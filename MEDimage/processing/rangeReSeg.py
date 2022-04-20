#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy

from numpy import ndarray


def rangeReSeg(vol, roi, im_range=None) -> ndarray:
    """Removs voxels from the intensity mask that fall outside
    the given range (intensities outside the range are set to 0).

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0's and 1's.
        im_range (ndarray): 1-D array with shape (1,2) of the 
        re-segmentation intensity range.

    Returns:
        ndarray: Intensity mask with intensities within the re-segmentation
            range.
    """

    if im_range is not None and len(im_range) == 2:
        roi = deepcopy(roi)
        roi[vol < im_range[0]] = 0
        roi[vol > im_range[1]] = 0

    return roi
