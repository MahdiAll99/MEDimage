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

"""Extraction of a particular features"""
    
def sde(gldzm):
    """
    Computes small distance emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(gldzm, 0)  # Distance Zone Vector

    # Small distance emphasis
    gldzm_sde = (np.matmul(pd, np.transpose(np.power(1.0 / np.array(c_vect), 2))))
    print(f"Fdzm_sde, {gldzm_sde}")

def lde(gldzm):
    """
    Computes large distance emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    pd = np.sum(gldzm, 0)  # Distance Zone Vector

    #Large distance emphasis
    return (np.matmul(pd, np.transpose(np.power(
        np.array(c_vect), 2))))


def lgze(gldzm):
    """
    Computes distance matrix low grey level zone emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Low grey level zone emphasisphasis
    return np.matmul(pg, np.transpose(np.power(
        1.0/np.array(r_vect), 2)))

def hgze(gldzm):
    """
    Computes distance matrix high grey level zone emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Low grey level zone emphasisphasis
    return np.matmul(pg, np.transpose(np.power(
        np.array(r_vect), 2)))

def sdlge(gldzm):
    """
    Computes small distance low grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Low grey level zone emphasisphasis
    return np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))


def sdhge(gldzm):
    """
    Computes small distance high grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #High grey level zone emphasisphasis
    return np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(1.0/c_mat, 2))))


def ldlge(gldzm):
    """
    Computes large distance low grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Large distance low grey levels emphasis
    return np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(c_mat, 2))))


def ldhge(gldzm):
    """
    Computes large distance high grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Large distance high grey levels emphasis
    return np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(c_mat, 2))))


def glnu(gldzm):
    """
    Computes distance zone matrix gray level non-uniformity
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    pg = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector
    ns = np.sum(gldzm)

    #Gray level non-uniformity
    return np.sum(np.power(pg, 2)) * ns


def glnu_norm(gldzm):
    """
    Computes distance zone matrix gray level non-uniformity normalised
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    pg = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Gray level non-uniformity normalised
    return np.sum(np.power(pg, 2))


def zdnu(gldzm):
    """
    Computes zone distance non-uniformity
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    pd = np.sum(gldzm, 0)  # Distance Zone Vector
    ns = np.sum(gldzm)

    #Zone distance non-uniformity
    return np.sum(np.power(pd, 2)) * ns


def zdnu_norm(gldzm):
    """
    Computes zone distance non-uniformity normalised
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    pd = np.sum(gldzm, 0)  # Distance Zone Vector

    #Zone distance non-uniformity normalised
    return np.sum(np.power(pd, 2))


def z_perc(gldzm, vol_int):
    """
    Computes zone percentage
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    ns = np.sum(gldzm)

    #Zone percentage
    return ns/np.sum(~np.isnan(vol_int[:]))


def gl_var(gldzm):
    """
    Computes grey level variance
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm
    temp = r_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * gldzm

    #Grey level variance
    return np.sum(temp)

def zd_var(gldzm):
    """
    Computes zone distance variance
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    sz = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm
    temp = c_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * gldzm

    #Zone distance variance
    return np.sum(temp)


def zd_entr(gldzm):
    """
    Computes zone distance entropy
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    val_pos = gldzm[np.nonzero(gldzm)]
    temp = val_pos * np.log2(val_pos)

    #Zone distance entropy
    return -np.sum(temp)


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
    levels = np.arange(1, np.max(vol_int[~np.isnan(vol_int[:])])+1)

    # GET THE gldzm MATRIX
    gldzm = get_gldzm_matrix(vol_int, mask_morph, levels)
    ns = np.sum(gldzm)
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
    gldzm['Fdzm_sdlge'] = np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Small distance high grey level emphasis
    gldzm['Fdzm_sdhge'] = np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Large distance low grey level emphasis
    gldzm['Fdzm_ldlge'] = np.sum(np.sum(gldzm*(np.power(
        1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # Large distance high grey level emphasis
    gldzm['Fdzm_ldhge'] = np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    gldzm['Fdzm_glnu'] = np.sum(np.power(pg, 2)) * ns

    # Gray level non-uniformity normalised
    gldzm['Fdzm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone distance non-uniformity
    gldzm['Fdzm_zdnu'] = np.sum(np.power(pd, 2)) * ns

    # Zone distance non-uniformity normalised
    gldzm['Fdzm_zdnu_norm'] = np.sum(np.power(pd, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm['Fdzm_z_perc'] = ns/np.sum(~np.isnan(vol_int[:]))

    # Grey level variance
    temp = r_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * gldzm
    gldzm['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = c_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * gldzm
    gldzm['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    val_pos = gldzm[np.nonzero(gldzm)]
    temp = val_pos * np.log2(val_pos)
    gldzm['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm
