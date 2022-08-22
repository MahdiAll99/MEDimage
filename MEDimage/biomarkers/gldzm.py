#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict, List, Union

import numpy as np
import scipy.ndimage as sc
import skimage.measure as skim


def get_matrix(roi_only_int: np.ndarray,
                     mask: np.ndarray,
                     levels: Union[np.ndarray, List]) -> np.ndarray:
    r"""
    Computes Grey level distance zone matrix.
    This matrix refers to "Grey level distance zone based features" (ID = VMDZ)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_. 

    Args:
        roi_only_int (ndarray): 3D volume, isotropically resampled,
            quantized (e.g. n_g = 32, levels = [1, ..., n_g]),
            with NaNs outside the region of interest.
        mask (ndarray): Morphological ROI ``mask``.
        levels (ndarray or List): Vector containing the quantized gray-levels
                                  in the tumor region (or reconstruction ``levels`` of quantization).

    Returns:
        ndarray: Grey level distance zone Matrix.

    Todo:
        ``levels`` should be removed at some point, no longer needed if we always 
        quantize our volume such that ``levels = 1,2,3,4,...,max(quantized Volume)``.
        So simply calculate ``levels = 1:max(roi_only(~isnan(roi_only(:))))``
        directly in this function.

    """

    roi_only_int = roi_only_int.copy()
    levels = levels.copy().astype("int")
    morph_voxel_grid = mask.copy().astype(np.uint8)

    # COMPUTATION OF DISTANCE MAP
    morph_voxel_grid = np.pad(morph_voxel_grid,
                            [1,1],
                            'constant',
                            constant_values=0)

    # Computing the smallest ROI edge possible.
    # Distances are determined in 3D
    binary_struct = sc.generate_binary_structure(rank=3, connectivity=1)
    perimeter = morph_voxel_grid - sc.binary_erosion(morph_voxel_grid, structure=binary_struct)
    perimeter = perimeter[1:-1,1:-1,1:-1] # Removing the padding.
    morph_voxel_grid = morph_voxel_grid[1:-1,1:-1,1:-1] # Removing the padding

    # +1 according to the definition of the IBSI    
    dist_map = sc.distance_transform_cdt(np.logical_not(perimeter), metric='cityblock') + 1

    # INITIALIZATION
    # Since levels is always defined as 1,2,3,4,...,max(quantized Volume)
    n_g = np.size(levels)
    level_temp = np.max(levels) + 1
    roi_only_int[np.isnan(roi_only_int)] = level_temp
    # Since the ROI morph always encompasses ROI int,
    # using the mask as defined from ROI morph does not matter since
    # we want to find the maximal possible distance.
    dist_init = np.max(dist_map[morph_voxel_grid == 1])
    gldzm = np.zeros((n_g,dist_init))

    # COMPUTATION OF gldzm
    temp = roi_only_int.copy().astype('int')
    for i in range(1,n_g+1):
        temp[roi_only_int!=levels[i-1]] = 0
        temp[roi_only_int==levels[i-1]] = 1
        conn_objects, n_zone = skim.label(temp,return_num = True)
        for j in range(1,n_zone+1):
            col = np.min(dist_map[conn_objects==j]).astype("int")
            gldzm[i-1,col-1] = gldzm[i-1,col-1] + 1

    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(gldzm,0))[0][-1]
    gldzm = np.delete(gldzm, range(stop+1, np.shape(gldzm)[1]), 1)

    return  gldzm

def extract_all(vol_int: np.ndarray,
                mask_morph: np.ndarray,
                gldzm: np.ndarray = None) -> Dict:
    """Computes gldzm features.
    This feature refers to "Grey level distance zone based features" (ID = VMDZ)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol_int (np.ndarray): 3D volume, isotropically resampled, quantized (e.g. n_g = 32, levels = [1, ..., n_g]),
        with NaNs outside the region of interest.
        mask_morph (np.ndarray): Morphological ROI mask.
        gldzm (np.ndarray, optional): array of the gray level distance zone matrix. Defaults to None.

    Returns:
        Dict: dict of ``gldzm`` features
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
    if gldzm is None:
        gldzm = get_matrix(vol_int, mask_morph, levels)
    n_s = np.sum(gldzm)
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    # Column and row indicators for each entry of the gldzm
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)
    p_g = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector
    p_d = np.sum(gldzm, 0)  # Distance Zone Vector

    # COMPUTING TEXTURES

    # Small distance emphasis
    gldzm_features['Fdzm_sde'] = (np.matmul(p_d, np.transpose(np.power(1.0/np.array(c_vect), 2))))

    # Large distance emphasis
    gldzm_features['Fdzm_lde'] = (np.matmul(p_d, np.transpose(np.power(np.array(c_vect), 2))))

    # Low grey level zone emphasis
    gldzm_features['Fdzm_lgze'] = np.matmul(p_g, np.transpose(np.power(1.0/np.array(r_vect), 2)))

    # High grey level zone emphasis
    gldzm_features['Fdzm_hgze'] = np.matmul(p_g, np.transpose(np.power(np.array(r_vect), 2)))

    # Small distance low grey level emphasis
    gldzm_features['Fdzm_sdlge'] = np.sum(np.sum(gldzm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Small distance high grey level emphasis
    gldzm_features['Fdzm_sdhge'] = np.sum(np.sum(gldzm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

    # Large distance low grey level emphasis
    gldzm_features['Fdzm_ldlge'] = np.sum(np.sum(gldzm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

    # Large distance high grey level emphasis
    gldzm_features['Fdzm_ldhge'] = np.sum(np.sum(gldzm*(np.power(r_mat, 2))*(np.power(c_mat, 2))))

    # Gray level non-uniformity
    gldzm_features['Fdzm_glnu'] = np.sum(np.power(p_g, 2)) * n_s

    # Gray level non-uniformity normalised
    gldzm_features['Fdzm_glnu_norm'] = np.sum(np.power(p_g, 2))

    # Zone distance non-uniformity
    gldzm_features['Fdzm_zdnu'] = np.sum(np.power(p_d, 2)) * n_s

    # Zone distance non-uniformity normalised
    gldzm_features['Fdzm_zdnu_norm'] = np.sum(np.power(p_d, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm_features['Fdzm_z_perc'] = n_s / np.sum(~np.isnan(vol_int[:]))

    # Grey level variance
    temp = r_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(r_mat-u, 2)) * gldzm
    gldzm_features['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = c_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(c_mat-u, 2)) * gldzm
    temp = (np.power(c_mat - u, 2)) * gldzm
    gldzm_features['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    val_pos = gldzm[np.nonzero(gldzm)]
    temp = val_pos * np.log2(val_pos)
    gldzm_features['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm_features

def get_single_matrix(vol_int: np.ndarray, mask_morph: np.ndarray) -> np.ndarray:
    """Computes gray level distance zone matrix in order to compute the single features.

    Args:
        vol_int (ndarray): 3D volume, isotropically resampled,
                           quantized (e.g. n_g = 32, levels = [1, ..., n_g]),
                           with NaNs outside the region of interest.
        mask_morph (ndarray): Morphological ROI mask.

    Returns:
        ndarray: gldzm features.
    """
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol_int[~np.isnan(vol_int[:])])+1)

    # GET THE gldzm MATRIX
    gldzm = get_matrix(vol_int, mask_morph, levels)

    return gldzm

def sde(gldzm: np.ndarray) -> float:
    """Computes small distance emphasis feature.
    This feature refers to "Fdzm_sde" (ID = 0GBI) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.
    
    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the small distance emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    p_d = np.sum(gldzm, 0)  # Distance Zone Vector

    # Small distance emphasis
    return (np.matmul(p_d, np.transpose(np.power(1.0 / np.array(c_vect), 2))))

def lde(gldzm: np.ndarray) -> float:
    """Computes large distance emphasis feature.
    This feature refers to "Fdzm_lde" (ID = MB4I) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the large distance emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    p_d = np.sum(gldzm, 0)  # Distance Zone Vector

    #Large distance emphasis
    return (np.matmul(p_d, np.transpose(np.power(np.array(c_vect), 2))))

def lgze(gldzm: np.ndarray) -> float:
    """Computes distance matrix low grey level zone emphasis feature.
    This feature refers to "Fdzm_lgze" (ID = S1RA) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the low grey level zone emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    r_vect = range(1, s_z[0]+1)  # Column vectors
    p_g = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Low grey level zone emphasisphasis
    return np.matmul(p_g, np.transpose(np.power(1.0/np.array(r_vect), 2)))

def hgze(gldzm: np.ndarray) -> float:
    """Computes distance matrix high grey level zone emphasis feature.
    This feature refers to "Fdzm_hgze" (ID = K26C) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the high grey level zone emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    r_vect = range(1, s_z[0]+1)  # Column vectors
    p_g = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Low grey level zone emphasisphasis
    return np.matmul(p_g, np.transpose(np.power(np.array(r_vect), 2)))

def sdlge(gldzm: np.ndarray) -> float:
    """Computes small distance low grey level emphasis feature.
    This feature refers to "Fdzm_sdlge" (ID = RUVG) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the low grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Low grey level zone emphasisphasis
    return np.sum(np.sum(gldzm*(np.power(1.0/r_mat, 2))*(np.power(1.0/c_mat, 2))))

def sdhge(gldzm: np.ndarray) -> float:
    """Computes small distance high grey level emphasis feature.
    This feature refers to "Fdzm_sdhge" (ID = DKNJ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the distance high grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #High grey level zone emphasisphasis
    return np.sum(np.sum(gldzm*(np.power(r_mat, 2))*(np.power(1.0/c_mat, 2))))

def ldlge(gldzm: np.ndarray) -> float:
    """Computes large distance low grey level emphasis feature.
    This feature refers to "Fdzm_ldlge" (ID = A7WM) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the low grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Large distance low grey levels emphasis
    return np.sum(np.sum(gldzm*(np.power(1.0/r_mat, 2))*(np.power(c_mat, 2))))

def ldhge(gldzm: np.ndarray) -> float:
    """Computes large distance high grey level emphasis feature.
    This feature refers to "Fdzm_ldhge" (ID = KLTH) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the high grey level emphasis feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    c_mat, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm

    #Large distance high grey levels emphasis
    return np.sum(np.sum(gldzm*(np.power(
        r_mat, 2))*(np.power(c_mat, 2))))

def glnu(gldzm: np.ndarray) -> float:
    """Computes distance zone matrix gray level non-uniformity
    This feature refers to "Fdzm_glnu" (ID = VFT7) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the gray level non-uniformity feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    p_g = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector
    n_s = np.sum(gldzm)

    #Gray level non-uniformity
    return np.sum(np.power(p_g, 2)) * n_s

def glnu_norm(gldzm: np.ndarray) -> float:
    """Computes distance zone matrix gray level non-uniformity normalised
    This feature refers to "Fdzm_glnu_norm" (ID = 7HP3) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the gray level non-uniformity normalised feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    p_g = np.transpose(np.sum(gldzm, 1))  # Gray-Level Vector

    #Gray level non-uniformity normalised
    return np.sum(np.power(p_g, 2))

def zdnu(gldzm: np.ndarray) -> float:
    """Computes zone distance non-uniformity
    This feature refers to "Fdzm_zdnu" (ID = V294) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the zone distance non-uniformity feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    p_d = np.sum(gldzm, 0)  # Distance Zone Vector
    n_s = np.sum(gldzm)

    #Zone distance non-uniformity
    return np.sum(np.power(p_d, 2)) * n_s

def zdnu_norm(gldzm: np.ndarray) -> float:
    """Computes zone distance non-uniformity normalised
    This feature refers to "Fdzm_zdnu_norm" (ID = IATH) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the zone distance non-uniformity normalised feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    p_d = np.sum(gldzm, 0)  # Distance Zone Vector

    #Zone distance non-uniformity normalised
    return np.sum(np.power(p_d, 2))

def z_perc(gldzm, vol_int):
    """Computes zone percentage
    This feature refers to "Fdzm_z_perc" (ID = VIWW) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the zone percentage feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    n_s = np.sum(gldzm)

    #Zone percentage
    return n_s/np.sum(~np.isnan(vol_int[:]))

def gl_var(gldzm: np.ndarray) -> float:
    """Computes grey level variance
    This feature refers to "Fdzm_gl_var" (ID = QK93) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the grey level variance feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    _, r_mat = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm
    temp = r_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(r_mat-u, 2)) * gldzm

    #Grey level variance
    return np.sum(temp)

def zd_var(gldzm: np.ndarray) -> float:
    """Computes zone distance variance
    This feature refers to "Fdzm_zd_var" (ID = 7WT1) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the zone distance variance feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    s_z = np.shape(gldzm)  # Size of gldzm
    c_vect = range(1, s_z[1]+1)  # Row vectors
    r_vect = range(1, s_z[0]+1)  # Column vectors
    c_mat, _ = np.meshgrid(c_vect, r_vect)  # Column and row indicators for each entry of the gldzm
    temp = c_mat * gldzm
    u = np.sum(temp)
    temp = (np.power(c_mat-u, 2)) * gldzm

    #Zone distance variance
    return np.sum(temp)

def zd_entr(gldzm: np.ndarray) -> float:
    """Computes zone distance entropy
    This feature refers to "Fdzm_zd_entr" (ID = GBDU) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        gldzm (ndarray): array of the gray level distance zone matrix

    Returns:
        float: the zone distance entropy feature
    """
    gldzm = gldzm / np.sum(gldzm)  # Normalization of gldzm
    val_pos = gldzm[np.nonzero(gldzm)]
    temp = val_pos * np.log2(val_pos)

    #Zone distance entropy
    return -np.sum(temp)
