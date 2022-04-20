#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def findIX(levels, fractVol, x) -> np.ndarray:
    """Computes intensity at volume fraction.

    Args:
        levels (ndarray): COMPLETE INTEGER grey-levels.
        fractVol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of largest volume fraction that has an 
            intensity fraction of at least `x`%.
    
    """
    ind = np.where(fractVol <= x/100)[0][0]
    Ix = levels[ind]

    return Ix
