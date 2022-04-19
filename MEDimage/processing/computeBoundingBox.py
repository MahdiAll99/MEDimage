#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def computeBoundingBox(mask) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box 
    in all dimensions.

    Args:
        mask (ndarray): ROI mask with values of 0 and 1.

    Returns:
        array: An array containing the indexes of the bounding box.

    """

    indices = np.where(np.reshape(mask, np.size(mask), order='F') == 1)
    iV, jV, kV = np.unravel_index(indices, np.shape(mask), order='F')
    boxBound = np.zeros((3, 2))
    boxBound[0, 0] = np.min(iV)
    boxBound[0, 1] = np.max(iV)
    boxBound[1, 0] = np.min(jV)
    boxBound[1, 1] = np.max(jV)
    boxBound[2, 0] = np.min(kV)
    boxBound[2, 1] = np.max(kV)

    return boxBound.astype(int)
