#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy

import numpy as np


def roi_extract(vol, roi) -> np.ndarray:
    """Replaces volume intensities outside the ROI with NaN.

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0's and 1's.

    Returns:
        ndarray: Imaging data with original intensities in the ROI 
            and NaN for intensities outside the ROI.
    """

    vol_re = deepcopy(vol)
    vol_re[roi == 0] = np.nan

    return vol_re
