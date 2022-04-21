#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np

from ..biomarkers.getGLSZMmatrix import getGLSZMmatrix


def getGLSZMfeatures(vol) -> Dict:
    """Computes GLSZM features.
    
    Args:
        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. Ng = 32, levels = [1, ..., Ng]),
            with NaNs outside the region of interest.
    
    Returns:
        Dict: Dict of GLSZM features.
        
    """

    glszm = {'Fszm_sze': [],
             'Fszm_lze': [],
             'Fszm_lgze': [],
             'Fszm_hgze': [],
             'Fszm_szlge': [],
             'Fszm_szhge': [],
             'Fszm_lzlge': [],
             'Fszm_lzhge': [],
             'Fszm_glnu': [],
             'Fszm_glnu_norm': [],
             'Fszm_zsnu': [],
             'Fszm_zsnu_norm': [],
             'Fszm_z_perc': [],
             'Fszm_gl_var': [],
             'Fszm_zs_var': [],
             'Fszm_zs_entr': []}

    # GET THE GLSZM MATRIX
    # Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)
    GLSZM = getGLSZMmatrix(vol, levels)
    Ns = np.sum(GLSZM)
    GLSZM = GLSZM/np.sum(GLSZM)  # Normalization of GLSZM
    sz = np.shape(GLSZM)  # Size of GLSZM

    cVect = range(1, sz[1]+1)  # Row vectors
    rVect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the GLSZM
    cMat, rMat = np.meshgrid(cVect, rVect)
    pg = np.transpose(np.sum(GLSZM, 1))  # Gray-Level Vector
    pz = np.sum(GLSZM, 0)  # Zone Size Vector

    # COMPUTING TEXTURES

    # Small zone emphasis
    glszm['Fszm_sze'] = (np.matmul(pz, np.transpose(np.power(1.0/np.array(cVect), 2))))

    # Large zone emphasis
    glszm['Fszm_lze'] = (np.matmul(pz, np.transpose(np.power(np.array(cVect), 2))))

    # Low grey level zone emphasis
    glszm['Fszm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(rVect), 2)))

    # High grey level zone emphasis
    glszm['Fszm_hgze'] = np.matmul(pg, np.transpose(np.power(np.array(rVect), 2)))

    # Small zone low grey level emphasis
    glszm['Fszm_szlge'] = np.sum(np.sum(GLSZM*(np.power(1.0/rMat, 2))*(np.power(1.0/cMat, 2))))

    # Small zone high grey level emphasis
    glszm['Fszm_szhge'] = np.sum(np.sum(GLSZM*(np.power(rMat, 2))*(np.power(1.0/cMat, 2))))

    # Large zone low grey levels emphasis
    glszm['Fszm_lzlge'] = np.sum(np.sum(GLSZM*(np.power(1.0/rMat, 2))*(np.power(cMat, 2))))

    # Large zone high grey level emphasis
    glszm['Fszm_lzhge'] = np.sum(np.sum(GLSZM*(np.power(rMat, 2))*(np.power(cMat, 2))))

    # Gray level non-uniformity
    glszm['Fszm_glnu'] = np.sum(np.power(pg, 2)) * Ns

    # Gray level non-uniformity normalised
    glszm['Fszm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone size non-uniformity
    glszm['Fszm_zsnu'] = np.sum(np.power(pz, 2)) * Ns

    # Zone size non-uniformity normalised
    glszm['Fszm_zsnu_norm'] = np.sum(np.power(pz, 2))

    # Zone percentage
    glszm['Fszm_z_perc'] = np.sum(pg)/(np.matmul(pz, np.transpose(cVect)))

    # Grey level variance
    temp = rMat * GLSZM
    u = np.sum(temp)
    temp = (np.power(rMat - u, 2)) * GLSZM
    glszm['Fszm_gl_var'] = np.sum(temp)

    # Zone size variance
    temp = cMat * GLSZM
    u = np.sum(temp)
    temp = (np.power(cMat - u, 2)) * GLSZM
    glszm['Fszm_zs_var'] = np.sum(temp)

    # Zone size entropy
    valPos = GLSZM[np.nonzero(GLSZM)]
    temp = valPos * np.log2(valPos)
    glszm['Fszm_zs_entr'] = -np.sum(temp)

    return glszm
