#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np


def outlierReSeg(vol, roi, outliers="") -> np.ndarray:
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

    TODO:
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
