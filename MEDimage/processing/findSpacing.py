#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from utils.mode import mode


def findSpacing(points, scanType) -> float:
    """
    Finds the slice spacing in mm.

    Note:
        This function works for points from at least 2 slices. If only
        one slice is present, the function returns a None.

    Args:
        points (ndarray): Array of (x,y,z) triplets defining a contour in the 
            Patient-Based Coordinate System extracted from DICOM RTstruct.
        scanType (str): Imaging modality (MRscan, CTscan...) 

    Returns:
        float: Slice spacing in mm.

    """
    decimKeep = 4  # We keep at most 4 decimals to find the slice spacing.

    # Rounding to the nearest 0.1 mm, MRI is more problematic due to arbitrary
    # orientations allowed for imaging volumes.
    if scanType == "MRscan":
        slices = np.unique(np.around(points, 1))
    else:
        slices = np.unique(np.around(points, 2))

    nSlices = len(slices)
    if nSlices == 1:
        return None

    diff = np.abs(np.diff(slices))
    diff = np.round(diff, decimKeep)
    sliceSpacing, nOcc = mode(x=diff, return_counts=True)
    if np.max(nOcc) == 1:
        sliceSpacing = np.mean(diff)

    return sliceSpacing
