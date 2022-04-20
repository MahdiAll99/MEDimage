#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def findVX(fractInt, fractVol, x) -> np.ndarray:
    """Computes volume at intensity fraction.

    Args:
        fractInt (ndarray): Intensity fraction.
        fractVol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of largest volume fraction `fractVol` that has an 
            intensity fraction `fractInt` of at least `x`%.

    """
    ind = np.where(fractInt >= x/100)[0][0]
    Vx = fractVol[ind]

    return Vx
