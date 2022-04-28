#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def getAreaDensApprox(a, b, c, n) -> float:
    """Computes area density - minimum volume enclosing ellipsoid
    
    Args:
        a (float): Major semi-axis length.
        b (float): Minor semi-axis length.
        c (float): Least semi-axis length.
        n (int): Number of iterations.

    Returns:
        float: Area density - minimum volume enclosing ellipsoid.

    """
    alpha = np.sqrt(1 - b**2/a**2)
    beta = np.sqrt(1 - c**2/a**2)
    AB = alpha * beta
    point = (alpha**2+beta**2) / (2*AB)
    Aell = 0

    for v in range(0, n+1):
        coef = [0]*v + [1]
        legen = np.polynomial.legendre.legval(x=point, c=coef)
        Aell = Aell + AB**v / (1-4*v**2) * legen

    Aell = Aell * 4 * np.pi * a * b

    return Aell
