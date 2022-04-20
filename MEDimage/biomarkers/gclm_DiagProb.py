#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gclm_DiagProb(p_ij) -> np.ndarray:
    """Computes diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighbouring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the diagonal probability.
    
    """

    Ng = np.size(p_ij, 0)
    valK = np.arange(0, Ng)
    nK = np.size(valK)
    p_iminusj = np.zeros(nK)

    for iterationK in range(0, nK):
        k = valK[iterationK]
        p = 0
        for i in range(0, Ng):
            for j in range(0, Ng):
                if (k - abs(i-j)) == 0:
                    p += p_ij[i, j]

        p_iminusj[iterationK] = p

    return p_iminusj
