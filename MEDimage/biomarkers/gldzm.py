#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

from ..biomarkers.getGLDZMmatrix import getGLDZMmatrix


def get_matrix(volInt, maskMorph):
    """
    Computes gray level distance zone matrix
    """
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(volInt[~np.isnan(volInt[:])])+1)

    # GET THE GLDZM MATRIX
    GLDZM = getGLDZMmatrix(volInt, maskMorph, levels)

    return GLDZM
    
def sde(GLDZM):
    """
    Computes distance zone matrix small distance emphasis feature
    """
    GLDZM = GLDZM / np.sum(GLDZM)  # Normalization of GLDZM
    sz = np.shape(GLDZM)  # Size of GLDZM
    cVect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(GLDZM, 0)  # Distance Zone Vector

    # Small distance emphasis
    return (np.matmul(pd, np.transpose(np.power(1.0 / np.array(cVect), 2))))

def extract_all(volInt, maskMorph) -> Dict:
    """Compute GLDZM features.
     
     Args:
        volInt (ndarray): 3D volume, isotropically resampled, 
            quantized (e.g. Ng = 32, levels = [1, ..., Ng]), 
            with NaNs outside the region of interest.
        maskMorph (ndarray): Morphological ROI mask.
    
    Returns:
        Dict: Dict of GLDZM features.

    """
    gldzm = {'Fdzm_sde': [],
             'Fdzm_lde': [],
             'Fdzm_lgze': [],
             'Fdzm_hgze': [],
             'Fdzm_sdlge': [],
             'Fdzm_sdhge': [],
             'Fdzm_ldlge': [],
             'Fdzm_ldhge': [],
             'Fdzm_glnu': [],
             'Fdzm_glnu_norm': [],
             'Fdzm_zdnu': [],
             'Fdzm_zdnu_norm': [],
             'Fdzm_z_perc': [],
             'Fdzm_gl_var': [],
             'Fdzm_zd_var': [],
             'Fdzm_zd_entr': []}

    # Correct definition, without any assumption
    levels = np.arange(1, np.max(volInt[~np.isnan(volInt[:])])+1)

    # GET THE GLDZM MATRIX
    GLDZM = getGLDZMmatrix(volInt, maskMorph, levels)
    Ns = np.sum(GLDZM)
    GLDZM = GLDZM/np.sum(GLDZM)  # Normalization of GLDZM
    sz = np.shape(GLDZM)  # Size of GLDZM
    cVect = range(1, sz[1]+1)  # Row vectors
    rVect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the GLDZM
    cMat, rMat = np.meshgrid(cVect, rVect)
    pg = np.transpose(np.sum(GLDZM, 1))  # Gray-Level Vector
    pd = np.sum(GLDZM, 0)  # Distance Zone Vector

    # COMPUTING TEXTURES

    # Small distance emphasis
    gldzm['Fdzm_sde'] = (np.matmul(pd, np.transpose(np.power(
        1.0/np.array(cVect), 2))))

    # Large distance emphasis
    gldzm['Fdzm_lde'] = (np.matmul(pd, np.transpose(np.power(
        np.array(cVect), 2))))

    # Low grey level zone emphasis
    gldzm['Fdzm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(rVect), 2)))

    # High grey level zone emphasis
    gldzm['Fdzm_hgze'] = np.matmul(pg, np.transpose(np.power(
        np.array(rVect), 2)))

    # Small distance low grey level emphasis
    gldzm['Fdzm_sdlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/rMat, 2))*(np.power(1.0/cMat, 2))))

    # Small distance high grey level emphasis
    gldzm['Fdzm_sdhge'] = np.sum(np.sum(GLDZM*(np.power(
        rMat, 2))*(np.power(1.0/cMat, 2))))

    # Large distance low grey levels emphasis
    gldzm['Fdzm_ldlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/rMat, 2))*(np.power(cMat, 2))))

    # Large distance high grey level emphasis
    gldzm['Fdzm_ldhge'] = np.sum(np.sum(GLDZM*(np.power(
        rMat, 2))*(np.power(cMat, 2))))

    # Gray level non-uniformity
    gldzm['Fdzm_glnu'] = np.sum(np.power(pg, 2)) * Ns

    # Gray level non-uniformity normalised
    gldzm['Fdzm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone distance non-uniformity
    gldzm['Fdzm_zdnu'] = np.sum(np.power(pd, 2)) * Ns

    # Zone distance non-uniformity normalised
    gldzm['Fdzm_zdnu_norm'] = np.sum(np.power(pd, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm['Fdzm_z_perc'] = Ns/np.sum(~np.isnan(volInt[:]))

    # Grey level variance
    temp = rMat * GLDZM
    u = np.sum(temp)
    temp = (np.power(rMat - u, 2)) * GLDZM
    gldzm['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = cMat * GLDZM
    u = np.sum(temp)
    temp = (np.power(cMat - u, 2)) * GLDZM
    gldzm['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    valPos = GLDZM[np.nonzero(GLDZM)]
    temp = valPos * np.log2(valPos)
    gldzm['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm
