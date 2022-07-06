#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

from ..biomarkers.get_gldzm_matrix import get_gldzm_matrix


def get_matrix(vol_int, mask_morph):
    """
    Computes gray level distance zone matrix
    """
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol_int[~np.isnan(vol_int[:])])+1)

    # GET THE gldzm MATRIX
    gldzm = get_gldzm_matrix(vol_int, mask_morph, levels)

    return gldzm
    
def sde(gldzm):
    """
    Computes distance zone matrix small distance emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(gldzm, 0)  # Distance Zone Vector

    # Small distance emphasis
    return (np.matmul(pd, np.transpose(np.power(1.0 / np.array(c_vect), 2))))

def extract_all(vol_int, mask_morph) -> Dict:
    """Compute gldzm features.
     
     Args:
        vol_int (ndarray): 3D volume, isotropically resampled, 
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]), 
            with NaNs outside the region of interest.
        mask_morph (ndarray): Morphological ROI mask.
    
    Returns:
        Dict: Dict of gldzm features.

    """
    gldzm_features = {'Fdzm_sde': [],
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
    levels = np.arange(1, np.max(vol_int[~np.isnan(vol_int[:])])+1)

    # GET THE gldzm MATRIX
    gldzm = get_gldzm_matrix(vol_int, mask_morph, levels)
    n_s = np.sum(gldzm)
    gldzm = gldzm/np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the gldzm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)
    pg = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector
    pd = np.sum(gldzm, 0)  # Distance Zone Vector

    # COMPUTING TEXTURES

    # Small distance emphasis
    gldzm_features['Fdzm_sde'] = (np.matmul(pd, np.transpose(np.power(
        1.0/np.array(c_vect), 2))))

    # Large distance emphasis
    gldzm_features['Fdzm_lde'] = (np.matmul(pd, np.transpose(np.power(
        np.array(c_vect), 2))))

    # Low grey level zone emphasis
    gldzm_features['Fdzm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(r_vect), 2)))

    # High grey level zone emphasis
    gldzm_features['Fdzm_hgze'] = np.matmul(pg, np.transpose(np.power(
        np.array(r_vect), 2)))

    # Small distance low grey level emphasis
    gldzm_features['Fdzm_sdlge'] = np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Small distance high grey level emphasis
    gldzm_features['Fdzm_sdhge'] = np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Large distance low grey levels emphasis
    gldzm_features['Fdzm_ldlge'] = np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # Large distance high grey level emphasis
    gldzm_features['Fdzm_ldhge'] = np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    gldzm_features['Fdzm_glnu'] = np.sum(np.power(pg, 2)) * n_s

    # Gray level non-uniformity normalised
    gldzm_features['Fdzm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone distance non-uniformity
    gldzm_features['Fdzm_zdnu'] = np.sum(np.power(pd, 2)) * n_s

    # Zone distance non-uniformity normalised
    gldzm_features['Fdzm_zdnu_norm'] = np.sum(np.power(pd, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm_features['Fdzm_z_perc'] = n_s/np.sum(~np.isnan(vol_int[:]))

    # Grey level variance
    temp = r_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * gldzm
    gldzm_features['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = c_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * gldzm
    gldzm_features['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    val_pos = gldzm[np.nonzero(gldzm)]
    temp = val_pos * np.log2(val_pos)
    gldzm_features['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm_features
