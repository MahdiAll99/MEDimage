#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np
from numpy import ndarray


def range_re_seg(vol: np.ndarray,
                 roi: np.ndarray,
                 im_range=None) -> ndarray:
    """Removes voxels from the intensity mask that fall outside
    the given range (intensities outside the range are set to 0).

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0's and 1's.
        im_range (ndarray): 1-D array with shape (1,2) of the re-segmentation intensity range.

    Returns:
        ndarray: Intensity mask with intensities within the re-segmentation range.
    """

    if im_range is not None and len(im_range) == 2:
        roi = deepcopy(roi)
        roi[vol < im_range[0]] = 0
        roi[vol > im_range[1]] = 0

    return roi

def outlier_re_seg(vol: np.ndarray,
                   roi: np.ndarray,
                   outliers="") -> np.ndarray:
    """Removes voxels with outlier intensities from the given mask
    using the Collewet method.

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0 and 1.
        outliers (str, optional): Algo used to define outliers.
                                  (For now this methods only implements "Collewet" method).

    Returns:
        ndarray: An array with values of 0 and 1.

    Raises:
        ValueError: If `outliers` is not "Collewet" or None.

    Todo:
        * Delete outliers argument or implements others outlining methods.
    """

    if outliers != '':
        roi = deepcopy(roi)

        if outliers == "Collewet":
            u = np.mean(vol[roi == 1])
            sigma = np.std(vol[roi == 1])

            roi[vol > (u + 3*sigma)] = 0
            roi[vol < (u - 3*sigma)] = 0
        else:
            raise ValueError("Outlier segmentation not defined.")

    return roi
