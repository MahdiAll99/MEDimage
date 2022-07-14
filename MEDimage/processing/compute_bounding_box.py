#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def compute_bounding_box(mask) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box 
    in all dimensions.

    Args:
        mask (ndarray): ROI mask with values of 0 and 1.

    Returns:
        ndarray: An array containing the indexes of the bounding box.
    """

    indices = np.where(np.reshape(mask, np.size(mask), order='F') == 1)
    iv, jv, kv = np.unravel_index(indices, np.shape(mask), order='F')
    box_bound = np.zeros((3, 2))
    box_bound[0, 0] = np.min(iv)
    box_bound[0, 1] = np.max(iv)
    box_bound[1, 0] = np.min(jv)
    box_bound[1, 1] = np.max(jv)
    box_bound[2, 0] = np.min(kv)
    box_bound[2, 1] = np.max(kv)

    return box_bound.astype(int)
