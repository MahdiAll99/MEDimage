#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np


def outlierReSeg(vol, roi) -> np.ndarray:
    """Removes voxels with outlier intensities from the given mask
    using the Collewet method.

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0 and 1.

    Returns:
        array: An array with values of 0 and 1.
    
    """

    roi = deepcopy(roi)

    u = np.mean(vol[roi == 1])
    sigma = np.std(vol[roi == 1])

    roi[vol > (u + 3*sigma)] = 0
    roi[vol < (u - 3*sigma)] = 0

    return roi
