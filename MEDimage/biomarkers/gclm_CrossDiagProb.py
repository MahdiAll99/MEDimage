#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def gclm_CrossDiagProb(p_ij) -> np.ndarray:
    """Computes cross diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighbouring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the cross diagonal probability.
     
    """
    Ng = np.size(p_ij, 0)
    valK = np.arange(2, 2*Ng + 100*np.finfo(float).eps)
    nK = np.size(valK)
    p_iplusj = np.zeros(nK)

    for iterationK in range(0, nK):
        k = valK[iterationK]
        p = 0
        for i in range(0, Ng):
            for j in range(0, Ng):
                if (k - (i+j+2)) == 0:
                    p += p_ij[i, j]

        p_iplusj[iterationK] = p

    return p_iplusj
