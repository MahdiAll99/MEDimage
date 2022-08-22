#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Union

import numpy as np
import skimage.measure as skim


def get_matrix(roi_only: np.ndarray, 
                     levels: Union[np.ndarray, List]) -> Dict:
    r"""
    This function computes the Gray-Level Size Zone Matrix (GLSZM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. The zones of different sizes are computed
    using 26-voxel connectivity.
    This matrix refers to "Grey level size zone based features" (ID = 9SAK)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_. 

    Note:
        This function is compatible with 2D analysis (language not adapted in the text).

    Args:
        roi_only_int (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels
            in the tumor region (or reconstruction ``levels`` of quantization).

    Returns:
        ndarray: Array of Gray-Level Size Zone Matrix of ``roi_only``.

    REFERENCES:
        [1] Thibault, G., Fertil, B., Navarro, C., Pereira, S., Cau, P., Levy,
        N., Mari, J.-L. (2009). Texture Indexes and Gray Level Size Zone
        Matrix. Application to Cell Nuclei Classification. In Pattern
        Recognition and Information Processing (PRIP) (pp. 140–145).
    """

    # PRELIMINARY
    roi_only = roi_only.copy()
    n_max = np.sum(~np.isnan(roi_only))
    level_temp = np.max(levels) + 1
    roi_only[np.isnan(roi_only)] = level_temp
    levels = np.append(levels, level_temp)

    # QUANTIZATION EFFECTS CORRECTION
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    unique_vect = levels
    n_l = np.size(levels) - 1

    # INITIALIZATION
    # THIS NEEDS TO BE CHANGED. THE ARRAY INITIALIZED COULD BE TOO BIG!
    glszm = np.zeros((n_l, n_max))

    # COMPUTATION OF glszm
    temp = roi_only.copy().astype('int')
    for i in range(1, n_l+1):
        temp[roi_only != unique_vect[i-1]] = 0
        temp[roi_only == unique_vect[i-1]] = 1
        conn_objects, n_zone = skim.label(temp, return_num=True)
        for j in range(1, n_zone+1):
            col = np.sum(conn_objects == j)
            glszm[i-1, col-1] = glszm[i-1, col-1] + 1

    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(glszm, 0))[0][-1]
    glszm = np.delete(glszm, range(stop+1, np.shape(glszm)[1]), 1)

    return glszm

def extract_all(vol: np.ndarray,
                glszm: np.ndarray = None) -> Dict:
    """Computes glszm features.
    These features refer to "Grey level size zone based features" (ID = 9SAK)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__. 
    
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

    # GET THE GLSZM MATRIX
    # Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)
    if glszm is None:
        glszm = get_matrix(vol, levels)
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

def get_single_matrix(vol: np.ndarray) -> np.ndarray:
    """Computes gray level size zone matrix in order to compute the single features.

    Args:
        vol_int: 3D volume, isotropically resampled, 
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]), 
            with NaNs outside the region of interest.
        levels: Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction ``levels`` of quantization).
    
    Returns:
        ndarray: Array of Gray-Level Size Zone Matrix of 'vol'.

    """
    # Correct definition, without any assumption
    vol = vol.copy()
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])])+1)

    # GET THE gldzm MATRIX
    glszm = get_matrix(vol, levels)

    return glszm

def sze(glszm: np.ndarray) -> float:
    """Computes small zone emphasis feature.
    This feature refers to "Fszm_sze" (ID = 5QRC) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_lze" (ID = 48P8) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_lgze" (ID = XMSY) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_hgze" (ID = 5GN9) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_szlge" (ID = 5RAI) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_szhge" (ID = HW1V) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_lzlge" (ID = YH51) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_lzhge" (ID = J17V) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    This feature refers to "Fszm_glnu" (ID = JNSA) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes grey level non-uniformity normalised 
    This feature refers to "Fszm_glnu_norm" (ID = Y1RO) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes zone size non-uniformity 
    This feature refers to "Fszm_zsnu" (ID = 4JP3) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes zone size non-uniformity normalised 
    This feature refers to "Fszm_zsnu_norm" (ID = VB3A) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes zone percentage 
    This feature refers to "Fszm_z_perc" (ID = P30P) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes grey level variance 
    This feature refers to "Fszm_gl_var" (ID = BYLV) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes zone size variance 
    This feature refers to "Fszm_zs_var" (ID = 3NSA) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
    """Computes zone size entropy 
    This feature refers to "Fszm_zs_entr" (ID = GU8N) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

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
