#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

import numpy as np

from ..biomarkers.get_glszm_matrix import get_glszm_matrix


def extract_all(vol: np.ndarray,
                glszm: np.ndarray = None) -> Dict:
    """Computes glszm features.
    
    Args:
        vol (ndarray): 3D volume, isotropically resampled, quantized
            (e.g. n_g = 32, levels = [1, ..., n_g]),
            with NaNs outside the region of interest.
    
    Returns:
        Dict: Dict of glszm features.
        
    """
    glszm_features = {'Fszm_sze': [],
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

    # GET THE glszm MATRIX
    # Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)
    if glszm is None:
        glszm = get_glszm_matrix(vol, levels)
    n_s = np.sum(glszm)
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)
    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector
    pz = np.sum(glszm, 0)  # Zone Size Vector

    # COMPUTING TEXTURES

    # Small zone emphasis
    glszm_features['Fszm_sze'] = (np.matmul(pz, np.transpose(np.power(1.0/np.array(c_vect), 2))))

    # Large zone emphasis
    glszm_features['Fszm_lze'] = (np.matmul(pz, np.transpose(np.power(np.array(c_vect), 2))))

    # Low grey level zone emphasis
    glszm_features['Fszm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(r_vect), 2)))

    # High grey level zone emphasis
    glszm_features['Fszm_hgze'] = np.matmul(pg, np.transpose(np.power(np.array(r_vect), 2)))

    # Small zone low grey level emphasis
    glszm_features['Fszm_szlge'] = np.sum(np.sum(glszm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Small zone high grey level emphasis
    glszm_features['Fszm_szhge'] = np.sum(np.sum(glszm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Large zone low grey levels emphasis
    glszm_features['Fszm_lzlge'] = np.sum(np.sum(glszm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # Large zone high grey level emphasis
    glszm_features['Fszm_lzhge'] = np.sum(np.sum(glszm*(np.power(r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    glszm_features['Fszm_glnu'] = np.sum(np.power(pg, 2)) * n_s

    # Gray level non-uniformity normalised
    glszm_features['Fszm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone size non-uniformity
    glszm_features['Fszm_zsnu'] = np.sum(np.power(pz, 2)) * n_s

    # Zone size non-uniformity normalised
    glszm_features['Fszm_zsnu_norm'] = np.sum(np.power(pz, 2))

    # Zone percentage
    glszm_features['Fszm_z_perc'] = np.sum(pg)/(np.matmul(pz, np.transpose(c_vect)))

    # Grey level variance
    temp = r_mat * glszm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * glszm
    glszm_features['Fszm_gl_var'] = np.sum(temp)

    # Zone size variance
    temp = c_mat * glszm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * glszm
    glszm_features['Fszm_zs_var'] = np.sum(temp)

    # Zone size entropy
    val_pos = glszm[np.nonzero(glszm)]
    temp = val_pos * np.log2(val_pos)
    glszm_features['Fszm_zs_entr'] = -np.sum(temp)

    return glszm_features

def get_matrix(vol: np.ndarray) -> np.ndarray:
    """Computes gray level size zone matrix.

    Args:
        vol_int: 3D volume, isotropically resampled, 
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]), 
            with NaNs outside the region of interest.
        levels: Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).
    
    Returns:
        ndarray: Array of Gray-Level Size Zone Matrix of 'vol'.

    """
    # Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)

    # GET THE gldzm MATRIX
    glszm = get_glszm_matrix(vol, levels)

    return glszm

def sze(glszm: np.ndarray) -> float:
    """Computes small zone emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the small zone emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    pz = np.sum(glszm, 0)  # Zone Size Vector

    # Small zone emphasis
    return (np.matmul(pz, np.transpose(np.power(1.0/np.array(c_vect), 2))))

def lze(glszm: np.ndarray) -> float:
    """Computes large zone emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the large zone emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    pz = np.sum(glszm, 0)  # Zone Size Vector

    # Large zone emphasis
    return (np.matmul(pz, np.transpose(np.power(np.array(c_vect), 2))))

def lgze(glszm: np.ndarray) -> float:
    """Computes low grey zone emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the low grey zone emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector

    # Low grey zone emphasis
    return np.matmul(pg, np.transpose(np.power(
        1.0/np.array(r_vect), 2)))

def hgze(glszm: np.ndarray) -> float:
    """Computes high grey zone emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the high grey zone emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    r_vect = range(1, sz[0]+1)  # Column vectors
    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector

    # High grey zone emphasis
    return np.matmul(pg, np.transpose(np.power(np.array(r_vect), 2)))

def szlge(glszm: np.ndarray) -> float:
    """Computes small zone low grey level emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the small zone low grey level emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    # Small zone low grey level emphasis
    return np.sum(np.sum(glszm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

def szhge(glszm: np.ndarray) -> float:
    """Computes small zone high grey level emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the small zone high grey level emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    # Small zone high grey level emphasis
    return np.sum(np.sum(glszm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

def lzlge(glszm: np.ndarray) -> float:
    """Computes large zone low grey level emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the large zone low grey level emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    # Lage zone low grey level emphasis
    return np.sum(np.sum(glszm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

def lzhge(glszm: np.ndarray) -> float:
    """Computes large zone high grey level emphasis feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the large zone high grey level emphasis

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)

    # Large zone high grey level emphasis
    return np.sum(np.sum(glszm*(np.power(r_mat, 2))*(np.power(c_mat, 2))))

def glnu(glszm: np.ndarray) -> float:
    """Computes grey level non-uniformity feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the grey level non-uniformity feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    n_s = np.sum(glszm)

    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector

    # Grey level non-uniformity feature
    return np.sum(np.power(pg, 2)) * n_s

def glnu_norm(glszm: np.ndarray) -> float:
    """Computes grey level non-uniformity normalised feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the grey level non-uniformity normalised feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm

    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector

    # Grey level non-uniformity normalised feature
    return np.sum(np.power(pg, 2))

def zsnu(glszm: np.ndarray) -> float:
    """Computes zone size non-uniformity feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the zone size non-uniformity feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    n_s = np.sum(glszm)

    pz = np.sum(glszm, 0)  # Zone Size Vector

    # Zone size non-uniformity feature
    return np.sum(np.power(pz, 2)) * n_s

def zsnu_norm(glszm: np.ndarray) -> float:
    """Computes zone size non-uniformity normalised feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the zone size non-uniformity normalised feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm

    pz = np.sum(glszm, 0)  # Zone Size Vector

    # Zone size non-uniformity normalised feature
    return np.sum(np.power(pz, 2))

def z_perc(glszm: np.ndarray) -> float:
    """Computes zone percentage feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the zone percentage feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    pg = np.transpose(np.sum(glszm, 1))  # Gray-Level Vector
    pz = np.sum(glszm, 0)  # Zone Size Vector

    # Zone percentage feature
    return np.sum(pg)/(np.matmul(pz, np.transpose(c_vect)))

def gl_var(glszm: np.ndarray) -> float:
    """Computes grey level variance feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the grey level variance feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    _, r_mat = np.meshgrid(c_vect, r_vect)

    temp = r_mat * glszm
    u = np.sum(temp)
    temp = (np.power(r_mat - u, 2)) * glszm

    # Grey level variance feature
    return np.sum(temp)

def zs_var(glszm: np.ndarray) -> float:
    """Computes zone size variance feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the zone size variance feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm
    sz = np.shape(glszm)  # Size of glszm

    c_vect = range(1, sz[1]+1)  # Row vectors
    r_vect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the glszm
    c_mat, _ = np.meshgrid(c_vect, r_vect)

    temp = c_mat * glszm
    u = np.sum(temp)
    temp = (np.power(c_mat - u, 2)) * glszm

    # Zone size variance feature
    return np.sum(temp)

def zs_entr(glszm: np.ndarray) -> float:
    """Computes zone size entropy feature.

    Args:
        glszm (ndarray): array of the gray level size zone matrix
    
    Returns:
        float: the zone size entropy feature

    """
    glszm = glszm/np.sum(glszm)  # Normalization of glszm

    val_pos = glszm[np.nonzero(glszm)]
    temp = val_pos * np.log2(val_pos)

    # Zone size entropy feature
    return -np.sum(temp)
