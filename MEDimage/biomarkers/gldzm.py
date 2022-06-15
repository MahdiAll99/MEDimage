#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

from ..biomarkers.get_gldzm_matrix import get_gldzm_matrix


def get_matrix(volInt, mask_morph):
    """
    Computes gray level distance zone matrix
    """
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(volInt[~np.isnan(volInt[:])])+1)

    # GET THE GLDZM MATRIX
    GLDZM = get_gldzm_matrix(volInt, mask_morph, levels)

    return GLDZM
    
def sde(GLDZM):
    """
    Computes distance zone matrix small distance emphasis feature
    """
    GLDZM = GLDZM / np.sum(GLDZM)  # Normalization of GLDZM
    sz = np.shape(GLDZM)  # Size of GLDZM
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(GLDZM, 0)  # Distance Zone Vector

    # Small distance emphasis
    return (np.matmul(pd, np.transpose(np.power(1.0 / np.array(c_vect), 2))))

def extract_all(volInt, mask_morph) -> Dict:
    """Compute GLDZM features.
     
     Args:
        volInt (ndarray): 3D volume, isotropically resampled, 
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]), 
            with NaNs outside the region of interest.
        mask_morph (ndarray): Morphological ROI mask.
    
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
    GLDZM = get_gldzm_matrix(volInt, mask_morph, levels)
    n_s = np.sum(GLDZM)
    GLDZM = GLDZM/np.sum(GLDZM)  # Normalization of GLDZM
    sz = np.shape(GLDZM)  # Size of GLDZM
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the GLDZM
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)
    pg = np.transpose(np.sum(GLDZM, 1))  # Gray-Level Vector
    pd = np.sum(GLDZM, 0)  # Distance Zone Vector

    # COMPUTING TEXTURES

    # Small distance emphasis
    gldzm['Fdzm_sde'] = (np.matmul(pd, np.transpose(np.power(
        1.0/np.array(c_vect), 2))))

    # Large distance emphasis
    gldzm['Fdzm_lde'] = (np.matmul(pd, np.transpose(np.power(
        np.array(c_vect), 2))))

    # Low grey level zone emphasis
    gldzm['Fdzm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(r_vect), 2)))

    # High grey level zone emphasis
    gldzm['Fdzm_hgze'] = np.matmul(pg, np.transpose(np.power(
        np.array(r_vect), 2)))

    # Small distance low grey level emphasis
    gldzm['Fdzm_sdlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Small distance high grey level emphasis
    gldzm['Fdzm_sdhge'] = np.sum(np.sum(GLDZM*(np.power(
        r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Large distance low grey levels emphasis
    gldzm['Fdzm_ldlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # Large distance high grey level emphasis
    gldzm['Fdzm_ldhge'] = np.sum(np.sum(GLDZM*(np.power(
        r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    gldzm['Fdzm_glnu'] = np.sum(np.power(pg, 2)) * n_s

    # Gray level non-uniformity normalised
    gldzm['Fdzm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone distance non-uniformity
    gldzm['Fdzm_zdnu'] = np.sum(np.power(pd, 2)) * n_s

    # Zone distance non-uniformity normalised
    gldzm['Fdzm_zdnu_norm'] = np.sum(np.power(pd, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm['Fdzm_z_perc'] = n_s/np.sum(~np.isnan(volInt[:]))

    # Grey level variance
    temp = r_mat * GLDZM
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * GLDZM
    gldzm['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = c_mat * GLDZM
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * GLDZM
    gldzm['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    val_pos = GLDZM[np.nonzero(GLDZM)]
    temp = val_pos * np.log2(val_pos)
    gldzm['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm
