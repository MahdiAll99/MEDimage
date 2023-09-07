#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Dict, List, Union, List

import numpy as np
import pandas as pd

from ..utils.textureTools import (coord2index, get_neighbour_direction,
                                  get_value, is_list_all_none)


def get_matrix(roi_only: np.ndarray,
                    levels: Union[np.ndarray, List],
                    dist_correction=True) -> np.ndarray:
    r"""
    This function computes the Gray-Level Co-occurence Matrix (GLCM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. Only one GLCM is computed per scan,
    simultaneously recording (i.e. adding up) the neighboring properties of
    the 26-connected neighbors of all voxels in the ROI. To account for
    discretization length differences, neighbors at a distance of :math:`\sqrt{3}`
    voxels around a center voxel increment the GLCM by a value of :math:`\sqrt{3}`,
    neighbors at a distance of :math:`\sqrt{2}` voxels around a center voxel increment
    the GLCM by a value of :math:`\sqrt{2}`, and neighbors at a distance of 1 voxels
    around a center voxel increment the GLCM by a value of 1.
    This matrix refers to "Grey level co-occurrence based features" (ID = LFYI) 
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        roi_only (ndarray): Smallest box containing the ROI, with the imaging data
                            ready for texture analysis computations. Voxels outside the ROI are
                            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels in the tumor region
                                  (or reconstruction ``levels`` of quantization).
        dist_correction (bool, optional): Set this variable to true in order to use
                                discretization length difference corrections as used by the `Institute of Physics and
                                Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`_.
                                Set this variable to false to replicate IBSI results.

    Returns:
        ndarray: Gray-Level Co-occurence Matrix of ``roi_only``.

    References:
        [1] Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural \
        features for image classification. IEEE Transactions on Systems, \
        Man and Cybernetics, smc 3(6), 610–621.
    """
    # PARSING "dist_correction" ARGUMENT
    if type(dist_correction) is not bool:
        # The user did not input either "true" or "false",
        # so the default behavior is used.
        dist_correction = True

    # PRELIMINARY
    roi_only = roi_only.copy()
    level_temp = np.max(levels)+1
    roi_only[np.isnan(roi_only)] = level_temp
    #levels = np.append(levels, level_temp)
    dim = np.shape(roi_only)

    if np.ndim(roi_only) == 2:
        dim[2] = 1

    q2 = np.reshape(roi_only, (1, np.prod(dim)))

    # QUANTIZATION EFFECTS CORRECTION (M. Vallieres)
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    # qs = round(levels*adjust)/adjust;
    # q2 = round(q2*adjust)/adjust;

    #qs = levels
    qs = levels.tolist() + [level_temp]
    lqs = np.size(qs)

    q3 = q2*0
    for k in range(0, lqs):
        q3[q2 == qs[k]] = k

    q3 = np.reshape(q3, dim).astype(int)
    GLCM = np.zeros((lqs, lqs))

    for i in range(1, dim[0]+1):
        i_min = max(1, i-1)
        i_max = min(i+1, dim[0])
        for j in range(1, dim[1]+1):
            j_min = max(1, j-1)
            j_max = min(j+1, dim[1])
            for k in range(1, dim[2]+1):
                k_min = max(1, k-1)
                k_max = min(k+1, dim[2])
                val_q3 = q3[i-1, j-1, k-1]
                for I2 in range(i_min, i_max+1):
                    for J2 in range(j_min, j_max+1):
                        for K2 in range(k_min, k_max+1):
                            if (I2 == i) & (J2 == j) & (K2 == k):
                                continue
                            else:
                                val_neighbor = q3[I2-1, J2-1, K2-1]
                                if dist_correction:
                                    # Discretization length correction
                                    GLCM[val_q3, val_neighbor] += \
                                        np.sqrt(abs(I2-i)+abs(J2-j)+abs(K2-k))
                                else:
                                    GLCM[val_q3, val_neighbor] += 1

    GLCM = GLCM[0:-1, 0:-1]

    return GLCM

def joint_max(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes joint maximum features.
    This feature refers to "Fcm_joint_max" (ID = GYBY) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]:: List or float of the joint maximum feature(s)
    """
    temp = []
    joint_max = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.max(df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, joint max: {sum(temp) / len(temp)}')
            joint_max.append(sum(temp) / len(temp))
    return joint_max

def joint_avg(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes joint  average features.
    This feature refers to "Fcm_joint_avg" (ID = 60VM) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]:: List or float of the joint  average feature(s)
    """
    temp = []
    joint_avg = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.i * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, joint avg: {sum(temp) / len(temp)}')
            joint_avg.append(sum(temp) / len(temp))
    return joint_avg

def joint_var(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes joint variance features.
    This feature refers to "Fcm_var" (ID = UR99) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: List or float of the joint variance feature(s)
    """
    temp = []
    joint_var = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pij.i * df_pij.pij)
            temp.append(np.sum((df_pij.i - m_u) ** 2.0 * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, joint var: {sum(temp) / len(temp)}')
            joint_var.append(sum(temp) / len(temp))
    return joint_var

def joint_entr(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes joint entropy features.
    This feature refers to "Fcm_joint_entr" (ID = TU9B) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the joint entropy features
    """
    temp = []
    joint_entr = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(-np.sum(df_pij.pij * np.log2(df_pij.pij)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, joint entr: {sum(temp) / len(temp)}')
            joint_entr.append(sum(temp) / len(temp))
    return joint_entr

def diff_avg(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes difference average features.
    This feature refers to "Fcm_diff_avg" (ID = TF7R) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the difference average features
    """
    temp = []
    diff_avg = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, df_pimj, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pimj.k * df_pimj.pimj))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, diff avg: {sum(temp) / len(temp)}')
            diff_avg.append(sum(temp) / len(temp))
    return diff_avg

def diff_var(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes difference variance features.
    This feature refers to "Fcm_diff_var" (ID = D3YU) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the difference variance features
    """
    temp = []
    diff_var = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, df_pimj, _, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pimj.k * df_pimj.pimj)
            temp.append(np.sum((df_pimj.k - m_u) ** 2.0 * df_pimj.pimj))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, diff var: {sum(temp) / len(temp)}')
            diff_var.append(sum(temp) / len(temp))
    return diff_var

def diff_entr(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes difference entropy features.
    This feature refers to "Fcm_diff_entr" (ID = NTRS) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the difference entropy features
    """
    temp = []
    diff_entr = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, df_pimj, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(-np.sum(df_pimj.pimj * np.log2(df_pimj.pimj)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, diff entr: {sum(temp) / len(temp)}')
            diff_entr.append(sum(temp) / len(temp))
    return diff_entr

def sum_avg(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes sum average features.
    This feature refers to "Fcm_sum_avg" (ID = ZGXS) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the sum average features
    """
    temp = []
    sum_avg = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, _, df_pipj, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pipj.k * df_pipj.pipj))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, sum avg: {sum(temp) / len(temp)}')
            sum_avg.append(sum(temp) / len(temp))
    return sum_avg

def sum_var(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes sum variance features.
    This feature refers to "Fcm_sum_var" (ID = OEEB) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the sum variance features
    """
    temp = []
    sum_var = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, _, df_pipj, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pipj.k * df_pipj.pipj)
            temp.append(np.sum((df_pipj.k - m_u) ** 2.0 * df_pipj.pipj))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, sum var: {sum(temp) / len(temp)}')
            sum_var.append(sum(temp) / len(temp))
    return sum_var

def sum_entr(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes sum entropy features.
    This feature refers to "Fcm_sum_entr" (ID = P6QZ) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the sum entropy features
    """
    temp = []
    sum_entr = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            _, _, _, _, df_pipj, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(-np.sum(df_pipj.pipj * np.log2(df_pipj.pipj)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, sum entr: {sum(temp) / len(temp)}')
            sum_entr.append(sum(temp) / len(temp))
    return sum_entr

def energy(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes angular second moment features.
    This feature refers to "Fcm_energy" (ID = 8ZQL) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the angular second moment features
    """
    temp = []
    energy = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.pij ** 2.0))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, energy: {sum(temp) / len(temp)}')
            energy.append(sum(temp) / len(temp))
    return energy

def contrast(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes constrast features.
    This feature refers to "Fcm_contrast" (ID = ACUI) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the contrast features
    """
    temp = []
    contrast = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum((df_pij.i - df_pij.j) ** 2.0 * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, contrast: {sum(temp) / len(temp)}')
            contrast.append(sum(temp) / len(temp))
    return contrast

def dissimilarity(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes dissimilarity features.
    This feature refers to "Fcm_dissimilarity" (ID = 8S9J) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the dissimilarity features
    """
    temp = []
    dissimilarity = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(np.abs(df_pij.i - df_pij.j) * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, dissimilarity: {sum(temp) / len(temp)}')
            dissimilarity.append(sum(temp) / len(temp))
    return dissimilarity

def inv_diff(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes inverse difference features.
    This feature refers to "Fcm_inv_diff" (ID = IB1Z) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the inverse difference features
    """
    temp = []
    inv_diff = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j))))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, inv diff: {sum(temp) / len(temp)}')
            inv_diff.append(sum(temp) / len(temp))
    return inv_diff

def inv_diff_norm(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes inverse difference normalized features.
    This feature refers to "Fcm_inv_diff_norm" (ID = NDRX) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the inverse difference normalized features
    """
    temp = []
    inv_diff_norm = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, n_g = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j) / n_g)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, inv diff norm: {sum(temp) / len(temp)}')
            inv_diff_norm.append(sum(temp) / len(temp))
    return inv_diff_norm

def inv_diff_mom(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes inverse difference moment features.
    This feature refers to "Fcm_inv_diff_mom" (ID = WF0Z) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the inverse difference moment features
    """
    temp = []
    inv_diff_mom = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, inv diff mom: {sum(temp) / len(temp)}')
            inv_diff_mom.append(sum(temp) / len(temp))
    return inv_diff_mom

def inv_diff_mom_norm(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes inverse difference moment normalized features.
    This feature refers to "Fcm_inv_diff_mom_norm" (ID = 1QCO) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the inverse difference moment normalized features
    """
    temp = []
    inv_diff_mom_norm = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, n_g = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j)** 2.0 / n_g ** 2.0)))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, inv diff mom norm: {sum(temp) / len(temp)}')
            inv_diff_mom_norm.append(sum(temp) / len(temp))
    return inv_diff_mom_norm

def inv_var(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes inverse variance features.
    This feature refers to "Fcm_inv_var" (ID = E8JP) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the inverse variance features
    """
    temp = []
    inv_var = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            mu_marg = np.sum(df_pi.i * df_pi.pi)
            var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)
            if var_marg == 0.0:
                temp.append(1.0)
            else:
                temp.append(1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, inv var: {sum(temp) / len(temp)}')
            inv_var.append(sum(temp) / len(temp))
    return inv_var

def corr(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes correlation features.
    This feature refers to "Fcm_corr" (ID = NI2N) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the correlation features
    """
    temp = []
    corr = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            mu_marg = np.sum(df_pi.i * df_pi.pi)
            var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)
            if var_marg == 0.0:
                temp.append(1.0)
            else:
                temp.append(1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, corr: {sum(temp) / len(temp)}')
            corr.append(sum(temp) / len(temp))
    return corr


def auto_corr(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes autocorrelation features.
    This feature refers to "Fcm_auto_corr" (ID = QWB0) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the autocorrelation features
    """
    temp = []
    auto_corr = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, _, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            temp.append(np.sum(df_pij.i * df_pij.j * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, auto corr: {sum(temp) / len(temp)}')
            auto_corr.append(sum(temp) / len(temp))
    return auto_corr

def info_corr1(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes information correlation 1 features.
    This feature refers to "Fcm_info_corr1" (ID = R8DG) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the information correlation 1 features
    """
    temp = []
    info_corr1 = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            hxy = -np.sum(df_pij.pij * np.log2(df_pij.pij))
            hxy_1 = -np.sum(df_pij.pij * np.log2(df_pij.pi * df_pij.pj))
            hx = -np.sum(df_pi.pi * np.log2(df_pi.pi))
            if len(df_pij) == 1 or hx == 0.0:
                temp.append(1.0)
            else:
                temp.append((hxy - hxy_1) / hx)
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, info corr 1: {sum(temp) / len(temp)}')
            info_corr1.append(sum(temp) / len(temp))
    return info_corr1

def info_corr2(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes information correlation 2 features - Note: iteration over combinations of i and j
    This feature refers to "Fcm_info_corr2" (ID = JN9H) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the information correlation 2 features
    """
    temp = []
    info_corr2 = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, df_pj, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
            hxy_2 = - np.sum(
                        np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)) * \
                        np.log2(np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)))
                        )
            if hxy_2 < hxy:
                temp.append(0)
            else:
                temp.append(np.sqrt(1 - np.exp(-2.0 * (hxy_2 - hxy))))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, info corr 2: {sum(temp) / len(temp)}')
            info_corr2.append(sum(temp) / len(temp))
    return info_corr2

def clust_tend(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes cluster tendency features.
    This feature refers to "Fcm_clust_tend" (ID = DG8W) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the cluster tendency features
    """
    temp = []
    clust_tend = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pi.i * df_pi.pi)
            temp.append(np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 2.0 * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, clust tend: {sum(temp) / len(temp)}')
            clust_tend.append(sum(temp) / len(temp))
    return clust_tend

def clust_shade(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes cluster shade features.
    This feature refers to "Fcm_clust_shade" (ID = 7NFM) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the cluster shade features
    """
    temp = []
    clust_shade = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pi.i * df_pi.pi)
            temp.append(np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 3.0 * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, clust shade: {sum(temp) / len(temp)}')
            clust_shade.append(sum(temp) / len(temp))
    return clust_shade

def clust_prom(glcm_dict: Dict) -> Union[float, List[float]]:
    """Computes cluster prominence features.
    This feature refers to "Fcm_clust_prom" (ID = AE86) in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        glcm_dict (Dict): dictionary with glcm matrices, generated using :func:`get_glcm_matrices`

    Returns:
        Union[float, List[float]]: the cluster prominence features
    """
    temp = []
    clust_prom = []
    for key in glcm_dict.keys():
        for glcm in glcm_dict[key]:
            df_pij, df_pi, _, _, _, _ = glcm.get_cm_data([np.nan, np.nan])
            m_u = np.sum(df_pi.i * df_pi.pi)
            temp.append(np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 4.0 * df_pij.pij))
        if len(glcm_dict) <= 1:
            return sum(temp) / len(temp)
        else:
            print(f'Merge method: {key}, clust prom: {sum(temp) / len(temp)}')
            clust_prom.append(sum(temp) / len(temp))
    return clust_prom

def extract_all(vol, dist_correction=None, merge_method="vol_merge") -> Dict:
    """Computes glcm features.
    This features refer to Glcm family in the `IBSI1 reference \
    manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, isotropically resampled, quantized
                       (e.g. n_g = 32, levels = [1, ..., n_g]), with NaNs outside the region
                       of interest.
        dist_correction (Union[bool, str], optional): Set this variable to true in order to use
                                                      discretization length difference corrections as used by the `Institute of Physics and
                                                      Engineering in Medicine <https://doi.org/10.1088/0031-9155/60/14/5471>`__.
                                                      Set this variable to false to replicate IBSI results.
                                                      Or use string and specify the norm for distance weighting. Weighting is
                                                      only performed if this argument is "manhattan", "euclidean" or "chebyshev".
        merge_method (str, optional): merging ``method`` which determines how features are
                                           calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
                                           Note that not all combinations of spatial and merge ``method`` are valid.
        method (str, optional): Either 'old' (deprecated) or 'new' (faster) ``method``.

    Returns:
        Dict: Dict of the glcm features.

    Raises:
        ValueError: If `method` is not 'old' or 'new'.

    Todo:

        - Enable calculation of CM features using different spatial methods (2d, 2.5d, 3d)
        - Enable calculation of CM features using different CM distance settings
        - Enable calculation of CM features for different merge methods ("average", "slice_merge", "dir_merge" and "vol_merge")
        - Provide the range of discretised intensities from a calling function and pass to :func:`get_cm_features`.
        - Test if dist_correction works as expected.

    """
    glcm = get_cm_features(
                        vol=vol,
                        intensity_range=[np.nan, np.nan],
                        merge_method=merge_method,
                        dist_weight_norm=dist_correction
                        )

    return glcm

def get_glcm_matrices(vol,
                    glcm_spatial_method="3d",
                    glcm_dist=1.0,
                    merge_method="vol_merge",
                    dist_weight_norm=None) -> Dict:
    """Extracts co-occurrence matrices from the intensity roi mask prior to features extraction.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.

    Args:
        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z).
        intensity_range (ndarray): range of potential discretised intensities,provided as a list: 
            [minimal discretised intensity, maximal discretised intensity]. 
            If one or both values are unknown, replace the respective values with np.nan.
        glcm_spatial_method (str, optional): spatial method which determines the way
            co-occurrence matrices are calculated and how features are determined.
            Must be "2d", "2.5d" or "3d".
        glcm_dist (float, optional): Chebyshev distance for comparison between neighboring voxels.
        merge_method (str, optional): merging method which determines how features are
            calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
            Note that not all combinations of spatial and merge method are valid.
        dist_weight_norm (Union[bool, str], optional): norm for distance weighting. Weighting is only
            performed if this argument is either "manhattan","euclidean", "chebyshev" or bool.

    Returns:
        Dict: Dict of co-occurrence matrices.

    Raises:
        ValueError: If `glcm_spatial_method` is not "2d", "2.5d" or "3d".
    """
    if type(glcm_spatial_method) is not list:
        glcm_spatial_method = [glcm_spatial_method]

    if type(glcm_dist) is not list:
        glcm_dist = [glcm_dist]

    if type(merge_method) is not list:
        merge_method = [merge_method]

    if type(dist_weight_norm) is bool:
        if dist_weight_norm:
            dist_weight_norm = "euclidean"

    # Get the roi in tabular format
    img_dims = vol.shape
    index_id = np.arange(start=0, stop=vol.size)
    coords = np.unravel_index(indices=index_id, shape=img_dims)
    df_img = pd.DataFrame({"index_id": index_id,
                           "g": np.ravel(vol),
                           "x": coords[0],
                           "y": coords[1],
                           "z": coords[2],
                           "roi_int_mask": np.ravel(np.isfinite(vol))})

    # Iterate over spatial arrangements
    for ii_spatial in glcm_spatial_method:
        # Iterate over distances
        for ii_dist in glcm_dist:
            # Initiate list of glcm objects
            glcm_list = []
            # Perform 2D analysis
            if ii_spatial.lower() in ["2d", "2.5d"]:
                # Iterate over slices
                for ii_slice in np.arange(0, img_dims[2]):
                    # Get neighbour direction and iterate over neighbours
                    nbrs = get_neighbour_direction(
                        d=1,
                        distance="chebyshev",
                        centre=False,
                        complete=False,
                        dim3=False) * int(ii_dist)
                    for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                        # Add glcm matrices to list
                        glcm_list += [CooccurrenceMatrix(distance=int(ii_dist),
                                                        direction=nbrs[:, ii_direction],
                                                        direction_id=ii_direction,
                                                        spatial_method=ii_spatial.lower(),
                                                        img_slice=ii_slice)]

            # Perform 3D analysis
            elif ii_spatial.lower() == "3d":
                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_direction(d=1,
                                            distance="chebyshev",
                                            centre=False,
                                            complete=False,
                                            dim3=True) * int(ii_dist)

                for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                    # Add glcm matrices to list
                    glcm_list += [CooccurrenceMatrix(distance=int(ii_dist), 
                                                    direction=nbrs[:, ii_direction], 
                                                    direction_id=ii_direction,
                                                    spatial_method=ii_spatial.lower())]

            else:
                raise ValueError(
                    "GCLM matrices can be determined in \"2d\", \"2.5d\" and \"3d\". \
                        The requested method (%s) is not implemented.", ii_spatial)

            # Calculate glcm matrices
            for glcm in glcm_list:
                glcm.calculate_cm_matrix(
                    df_img=df_img, img_dims=img_dims, dist_weight_norm=dist_weight_norm)

            # Merge matrices according to the given method
            upd_list = {}
            for merge_method in merge_method:
                upd_list[merge_method] = combine_matrices(
                    glcm_list=glcm_list, merge_method=merge_method, spatial_method=ii_spatial.lower())

                # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
                if upd_list is None:
                    continue
    return upd_list

def get_cm_features(vol,
                    intensity_range,
                    glcm_spatial_method="3d",
                    glcm_dist=1.0,
                    merge_method="vol_merge",
                    dist_weight_norm=None) -> Dict:
    """Extracts co-occurrence matrix-based features from the intensity roi mask.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.

    Args:
        vol (ndarray): volume with discretised intensities as 3D numpy array (x, y, z).
        intensity_range (ndarray): range of potential discretised intensities,
                                   provided as a list: [minimal discretised intensity, maximal discretised
                                   intensity]. If one or both values are unknown, replace the respective values
                                   with np.nan.
        glcm_spatial_method (str, optional): spatial method which determines the way
                                             co-occurrence matrices are calculated and how features are determined.
                                             MUST BE "2d", "2.5d" or "3d".
        glcm_dist (float, optional): chebyshev distance for comparison between neighbouring
                                     voxels.
        merge_method (str, optional): merging method which determines how features are
                                           calculated. One of "average", "slice_merge", "dir_merge" and "vol_merge".
                                           Note that not all combinations of spatial and merge method are valid.
        dist_weight_norm (Union[bool, str], optional): norm for distance weighting. Weighting is only
                                                       performed if this argument is either "manhattan",
                                                       "euclidean", "chebyshev" or bool.

    Returns:
        Dict: Dict of the glcm features.

    Raises:
        ValueError: If `glcm_spatial_method` is not "2d", "2.5d" or "3d".
    """
    if type(glcm_spatial_method) is not list:
        glcm_spatial_method = [glcm_spatial_method]

    if type(glcm_dist) is not list:
        glcm_dist = [glcm_dist]

    if type(merge_method) is not list:
        merge_method = [merge_method]

    if type(dist_weight_norm) is bool:
        if dist_weight_norm:
            dist_weight_norm = "euclidean"

    # Get the roi in tabular format
    img_dims = vol.shape
    index_id = np.arange(start=0, stop=vol.size)
    coords = np.unravel_index(indices=index_id, shape=img_dims)
    df_img = pd.DataFrame({"index_id": index_id,
                           "g": np.ravel(vol),
                           "x": coords[0],
                           "y": coords[1],
                           "z": coords[2],
                           "roi_int_mask": np.ravel(np.isfinite(vol))})

    # Generate an empty feature list
    feat_list = []

    # Iterate over spatial arrangements
    for ii_spatial in glcm_spatial_method:
        # Iterate over distances
        for ii_dist in glcm_dist:
            # Initiate list of glcm objects
            glcm_list = []
            # Perform 2D analysis
            if ii_spatial.lower() in ["2d", "2.5d"]:
                # Iterate over slices
                for ii_slice in np.arange(0, img_dims[2]):
                    # Get neighbour direction and iterate over neighbours
                    nbrs = get_neighbour_direction(
                        d=1,
                        distance="chebyshev",
                        centre=False,
                        complete=False,
                        dim3=False) * int(ii_dist)
                    for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                        # Add glcm matrices to list
                        glcm_list += [CooccurrenceMatrix(distance=int(ii_dist),
                                                        direction=nbrs[:, ii_direction],
                                                        direction_id=ii_direction,
                                                        spatial_method=ii_spatial.lower(),
                                                        img_slice=ii_slice)]

            # Perform 3D analysis
            elif ii_spatial.lower() == "3d":
                # Get neighbour direction and iterate over neighbours
                nbrs = get_neighbour_direction(d=1,
                                            distance="chebyshev",
                                            centre=False,
                                            complete=False,
                                            dim3=True) * int(ii_dist)

                for ii_direction in np.arange(0, np.shape(nbrs)[1]):
                    # Add glcm matrices to list
                    glcm_list += [CooccurrenceMatrix(distance=int(ii_dist), 
                                                    direction=nbrs[:, ii_direction], 
                                                    direction_id=ii_direction,
                                                    spatial_method=ii_spatial.lower())]

            else:
                raise ValueError(
                    "GCLM matrices can be determined in \"2d\", \"2.5d\" and \"3d\". \
                        The requested method (%s) is not implemented.", ii_spatial)

            # Calculate glcm matrices
            for glcm in glcm_list:
                glcm.calculate_cm_matrix(
                    df_img=df_img, img_dims=img_dims, dist_weight_norm=dist_weight_norm)

            # Merge matrices according to the given method
            for merge_method in merge_method:
                upd_list = combine_matrices(
                    glcm_list=glcm_list, merge_method=merge_method, spatial_method=ii_spatial.lower())

                # Skip if no matrices are available (due to illegal combinations of merge and spatial methods
                if upd_list is None:
                    continue

                # Calculate features
                feat_run_list = []
                for glcm in upd_list:
                    feat_run_list += [glcm.calculate_cm_features(
                        intensity_range=intensity_range)]

                # Average feature values
                feat_list += [pd.concat(feat_run_list, axis=0).mean(axis=0, skipna=True).to_frame().transpose()]

    # Merge feature tables into a single table and return as a dictionary
    df_feat = pd.concat(feat_list, axis=1).to_dict(orient="records")[0]

    return df_feat

def combine_matrices(glcm_list: List, merge_method: str, spatial_method: str) -> List:
    """Merges co-occurrence matrices prior to feature calculation.

    Note:
        This code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.

    Args:
        glcm_list (List): List of CooccurrenceMatrix objects.
        merge_method (str): Merging method which determines how features are calculated.
                            One of "average", "slice_merge", "dir_merge" and "vol_merge". Note that not all
                            combinations of spatial and merge method are valid.
        spatial_method (str): spatial method which determines the way co-occurrence
                              matrices are calculated and how features are determined. One of "2d", "2.5d"
                              or "3d".

    Returns:
        List[CooccurrenceMatrix]: list of one or more merged CooccurrenceMatrix objects.
    """
    # Initiate empty list
    use_list = []

    # For average features over direction, maintain original glcms
    if merge_method == "average" and spatial_method in ["2d", "3d"]:
        # Make copy of glcm_list
        for glcm in glcm_list:
            use_list += [glcm._copy()]

        # Set merge method to average
        for glcm in use_list:
            glcm.merge_method = "average"

    # Merge glcms by slice
    elif merge_method == "slice_merge" and spatial_method == "2d":
        # Find slice_ids
        slice_id = []
        for glcm in glcm_list:
            slice_id += [glcm.slice]

        # Iterate over unique slice_ids
        for ii_slice in np.unique(slice_id):
            slice_glcm_id = np.squeeze(np.where(slice_id == ii_slice))

            # Select all matrices within the slice
            sel_matrix_list = []
            for glcm_id in slice_glcm_id:
                sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected slice
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance,
                                                direction=None,
                                                direction_id=None,
                                                spatial_method=spatial_method,
                                                img_slice=ii_slice,
                                                merge_method=merge_method,
                                                matrix=None,
                                                n_v=0.0)]
            else:
                # Merge matrices within the slice
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels within the merged slice
                merge_n_v = 0.0
                for glcm_id in slice_glcm_id:
                    merge_n_v += glcm_list[glcm_id].n_v

                # Create new cooccurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[slice_glcm_id[0]].distance,
                                                direction=None,
                                                direction_id=None,
                                                spatial_method=spatial_method,
                                                img_slice=ii_slice,
                                                merge_method=merge_method,
                                                matrix=merge_cm,
                                                n_v=merge_n_v)]

    # Merge glcms by direction
    elif merge_method == "dir_merge" and spatial_method == "2.5d":
        # Find slice_ids
        dir_id = []
        for glcm in glcm_list:
            dir_id += [glcm.direction_id]

        # Iterate over unique directions
        for ii_dir in np.unique(dir_id):
            dir_glcm_id = np.squeeze(np.where(dir_id == ii_dir))

            # Select all matrices with the same direction
            sel_matrix_list = []
            for glcm_id in dir_glcm_id:
                sel_matrix_list += [glcm_list[glcm_id].matrix]

            # Check if any matrix has been created for the currently selected direction
            if is_list_all_none(sel_matrix_list):
                # No matrix was created
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance,
                                                direction=glcm_list[dir_glcm_id[0]].direction,
                                                direction_id=ii_dir,
                                                spatial_method=spatial_method,
                                                img_slice=None,
                                                merge_method=merge_method,
                                                matrix=None, n_v=0.0)]
            else:
                # Merge matrices with the same direction
                merge_cm = pd.concat(sel_matrix_list, axis=0)
                merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

                # Update the number of voxels for the merged matrices with the same direction
                merge_n_v = 0.0
                for glcm_id in dir_glcm_id:
                    merge_n_v += glcm_list[glcm_id].n_v

                # Create new co-occurrence matrix
                use_list += [CooccurrenceMatrix(distance=glcm_list[dir_glcm_id[0]].distance,
                                                direction=glcm_list[dir_glcm_id[0]].direction,
                                                direction_id=ii_dir,
                                                spatial_method=spatial_method,
                                                img_slice=None,
                                                merge_method=merge_method,
                                                matrix=merge_cm,
                                                n_v=merge_n_v)]

    # Merge all glcms into a single representation
    elif merge_method == "vol_merge" and spatial_method in ["2.5d", "3d"]:
        # Select all matrices within the slice
        sel_matrix_list = []
        for glcm_id in np.arange(len(glcm_list)):
            sel_matrix_list += [glcm_list[glcm_id].matrix]

        # Check if any matrix was created
        if is_list_all_none(sel_matrix_list):
            # In case no matrix was created
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance,
                                            direction=None,
                                            direction_id=None,
                                            spatial_method=spatial_method,
                                            img_slice=None,
                                            merge_method=merge_method,
                                            matrix=None,
                                            n_v=0.0)]
        else:
            # Merge co-occurrence matrices
            merge_cm = pd.concat(sel_matrix_list, axis=0)
            merge_cm = merge_cm.groupby(by=["i", "j"]).sum().reset_index()

            # Update the number of voxels
            merge_n_v = 0.0
            for glcm_id in np.arange(len(glcm_list)):
                merge_n_v += glcm_list[glcm_id].n_v

            # Create new co-occurrence matrix
            use_list += [CooccurrenceMatrix(distance=glcm_list[0].distance,
                                            direction=None,
                                            direction_id=None,
                                            spatial_method=spatial_method,
                                            img_slice=None,
                                            merge_method=merge_method,
                                            matrix=merge_cm,
                                            n_v=merge_n_v)]
    else:
        use_list = None

    return use_list


class CooccurrenceMatrix:
    """ Class that contains a single co-occurrence ``matrix``.

    Note:
        Code was adapted from the in-house radiomics software created at
        OncoRay, Dresden, Germany.

    Attributes:
        distance (int): Chebyshev ``distance``.
        direction (ndarray): Direction along which neighbouring voxels are found.
        direction_id (int): Direction index to identify unique ``direction`` vectors.
        spatial_method (str): Spatial method used to calculate the co-occurrence
                              ``matrix``: "2d", "2.5d" or "3d".
        img_slice (ndarray): Corresponding slice index (only if the co-occurrence
                             ``matrix`` corresponds to a 2d image slice).
        merge_method (str): Method for merging the co-occurrence ``matrix`` with other
                            co-occurrence matrices.
        matrix (pandas.DataFrame): The actual co-occurrence ``matrix`` in sparse format
                                   (row, column, count).
        n_v (int): The number of voxels in the volume.
    """

    def __init__(self,
                distance: int,
                direction: np.ndarray,
                direction_id: int,
                spatial_method: str,
                img_slice: np.ndarray=None,
                merge_method: str=None,
                matrix: pd.DataFrame=None,
                n_v: int=None) -> None:
        """Constructor of the CooccurrenceMatrix class

        Args:
            distance (int): Chebyshev ``distance``.
            direction (ndarray): Direction along which neighbouring voxels are found.
            direction_id (int): Direction index to identify unique ``direction`` vectors.
            spatial_method (str): Spatial method used to calculate the co-occurrence
                                ``matrix``: "2d", "2.5d" or "3d".
            img_slice (ndarray, optional): Corresponding slice index (only if the
                                        co-occurrence ``matrix`` corresponds to a 2d image slice).
            merge_method (str, optional): Method for merging the co-occurrence ``matrix``
                                        with other co-occurrence matrices.
            matrix (pandas.DataFrame, optional): The actual co-occurrence ``matrix`` in
                                                sparse format (row, column, count).
            n_v (int, optional): The number of voxels in the volume.
        
        Returns:
            None.
        """
        # Distance used
        self.distance = distance

        # Direction and slice for which the current matrix is extracted
        self.direction = direction
        self.direction_id = direction_id
        self.img_slice = img_slice

        # Spatial analysis method (2d, 2.5d, 3d) and merge method (average, slice_merge, dir_merge, vol_merge)
        self.spatial_method = spatial_method
        self.merge_method = merge_method

        # Place holders
        self.matrix = matrix
        self.n_v = n_v

    def _copy(self):
        """
        Returns a copy of the co-occurrence matrix object.
        """
        return deepcopy(self)

    def calculate_cm_matrix(self, df_img: pd.DataFrame, img_dims: np.ndarray, dist_weight_norm: str) -> None:
        """Function that calculates a co-occurrence matrix for the settings provided during
        initialisation and the input image.

        Args:
            df_img (pandas.DataFrame): Data table containing image intensities, x, y and z coordinates,
                and mask labels corresponding to voxels in the volume.
            img_dims (ndarray, List[float]): Dimensions of the image volume.
            dist_weight_norm (str): Norm for distance weighting. Weighting is only
                performed if this parameter is either "manhattan", "euclidean" or "chebyshev".

        Returns:
            None. Assigns the created image table (cm matrix) to the `matrix` attribute.

        Raises:
            ValueError:
                If `self.spatial_method` is not "2d", "2.5d" or "3d".
                Also, if ``dist_weight_norm`` is not "manhattan", "euclidean" or "chebyshev".

        """
        # Check if the roi contains any masked voxels. If this is not the case, don't construct the glcm.
        if not np.any(df_img.roi_int_mask):
            self.n_v = 0
            self.matrix = None

            return None

        # Create local copies of the image table
        if self.spatial_method == "3d":
            df_cm = deepcopy(df_img)
        elif self.spatial_method in ["2d", "2.5d"]:
            df_cm = deepcopy(df_img[df_img.z == self.img_slice])
            df_cm["index_id"] = np.arange(0, len(df_cm))
            df_cm["z"] = 0
            df_cm = df_cm.reset_index(drop=True)
        else:
            raise ValueError(
                "The spatial method for grey level co-occurrence matrices should be one of \"2d\", \"2.5d\" or \"3d\".")

        # Set grey level of voxels outside ROI to NaN
        df_cm.loc[df_cm.roi_int_mask == False, "g"] = np.nan

        # Determine potential transitions
        df_cm["to_index"] = coord2index(x=df_cm.x.values + self.direction[0],
                                        y=df_cm.y.values + self.direction[1],
                                        z=df_cm.z.values + self.direction[2],
                                        dims=img_dims)

        # Get grey levels from transitions
        df_cm["to_g"] = get_value(x=df_cm.g.values, index=df_cm.to_index.values)

        # Check if any transitions exist.
        if np.all(np.isnan(df_cm[["to_g"]])):
            self.n_v = 0
            self.matrix = None

            return None

        # Count occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).size().reset_index(name="n")

        # Append grey level transitions in opposite direction
        df_cm_inv = pd.DataFrame({"g": df_cm.to_g, "to_g": df_cm.g, "n": df_cm.n})
        df_cm = df_cm.append(df_cm_inv, ignore_index=True)

        # Sum occurrences of grey level transitions
        df_cm = df_cm.groupby(by=["g", "to_g"]).sum().reset_index()

        # Rename columns
        df_cm.columns = ["i", "j", "n"]

        if dist_weight_norm in ["manhattan", "euclidean", "chebyshev"]:
            if dist_weight_norm == "manhattan":
                weight = sum(abs(self.direction))
            elif dist_weight_norm == "euclidean":
                weight = np.sqrt(sum(np.power(self.direction, 2.0)))
            elif dist_weight_norm == "chebyshev":
                weight = np.max(abs(self.direction))
            df_cm.n /= weight

        # Set the number of voxels
        self.n_v = np.sum(df_cm.n)

        # Add matrix and number of voxels to object
        self.matrix = df_cm

    def get_cm_data(self, intensity_range: np.ndarray):
        """Computes the probability distribution for the elements of the GLCM
        (diagonal probability, cross-diagonal probability...) and number of gray-levels.

        Args:
            intensity_range (ndarray): Range of potential discretised intensities, provided as a list:
                [minimal discretised intensity, maximal discretised intensity].
                If one or both values are unknown,replace the respective values with np.nan.

        Returns:
            Typle[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
            - Occurence data frame
            - Diagonal probabilty
            - Cross-diagonal probabilty
            - Number of gray levels
        """
        # Occurrence data frames
        df_pij = deepcopy(self.matrix)
        df_pij["pij"] = df_pij.n / sum(df_pij.n)
        df_pi = df_pij.groupby(by="i")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pi"})
        df_pj = df_pij.groupby(by="j")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pj"})

        # Diagonal probilities p(i-j)
        df_pimj = deepcopy(df_pij)
        df_pimj["k"] = np.abs(df_pimj.i - df_pimj.j)
        df_pimj = df_pimj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pimj"})

        # Cross-diagonal probabilities p(i+j)
        df_pipj = deepcopy(df_pij)
        df_pipj["k"] = df_pipj.i + df_pipj.j
        df_pipj = df_pipj.groupby(by="k")["pij"].agg(np.sum).reset_index().rename(columns={"pij": "pipj"})

        # Merger of df.p_ij, df.p_i and df.p_j
        df_pij = pd.merge(df_pij, df_pi, on="i")
        df_pij = pd.merge(df_pij, df_pj, on="j")

        # Constant definitions
        intensity_range_loc = deepcopy(intensity_range)
        if np.isnan(intensity_range[0]):
            intensity_range_loc[0] = np.min(df_pi.i) * 1.0
        if np.isnan(intensity_range[1]):
            intensity_range_loc[1] = np.max(df_pi.i) * 1.0
        # Number of grey levels
        n_g = intensity_range_loc[1] - intensity_range_loc[0] + 1.0

        return df_pij, df_pi, df_pj, df_pimj, df_pipj, n_g

    def calculate_cm_features(self, intensity_range: np.ndarray) -> pd.DataFrame:
        """Wrapper to json.dump function.

        Args:
            intensity_range (np.ndarray): Range of potential discretised intensities, 
                provided as a list: [minimal discretised intensity, maximal discretised intensity].
                If one or both values are unknown,replace the respective values with np.nan.

        Returns:
            pandas.DataFrame: Data frame with values for each feature.
        """
        # Create feature table
        feat_names = ["Fcm_joint_max", "Fcm_joint_avg", "Fcm_joint_var", "Fcm_joint_entr",
                      "Fcm_diff_avg", "Fcm_diff_var", "Fcm_diff_entr",
                      "Fcm_sum_avg", "Fcm_sum_var", "Fcm_sum_entr",
                      "Fcm_energy", "Fcm_contrast", "Fcm_dissimilarity",
                      "Fcm_inv_diff", "Fcm_inv_diff_norm", "Fcm_inv_diff_mom",
                      "Fcm_inv_diff_mom_norm", "Fcm_inv_var", "Fcm_corr",
                      "Fcm_auto_corr", "Fcm_clust_tend", "Fcm_clust_shade",
                      "Fcm_clust_prom", "Fcm_info_corr1", "Fcm_info_corr2"]

        df_feat = pd.DataFrame(np.full(shape=(1, len(feat_names)), fill_value=np.nan))
        df_feat.columns = feat_names

        # Don't return data for empty slices or slices without a good matrix
        if self.matrix is None:
            # Update names
            #df_feat.columns += self._parse_names()
            return df_feat
        elif len(self.matrix) == 0:
            # Update names
            #df_feat.columns += self._parse_names()
            return df_feat

        df_pij, df_pi, df_pj, df_pimj, df_pipj, n_g = self.get_cm_data(intensity_range)

        ###############################################
        ######           glcm features           ######
        ###############################################
        # Joint maximum
        df_feat.loc[0, "Fcm_joint_max"] = np.max(df_pij.pij)

        # Joint average
        df_feat.loc[0, "Fcm_joint_avg"] = np.sum(df_pij.i * df_pij.pij)

        # Joint variance
        m_u = np.sum(df_pij.i * df_pij.pij)
        df_feat.loc[0, "Fcm_joint_var"] = np.sum((df_pij.i - m_u) ** 2.0 * df_pij.pij)

        # Joint entropy
        df_feat.loc[0, "Fcm_joint_entr"] = -np.sum(df_pij.pij * np.log2(df_pij.pij))

        # Difference average
        df_feat.loc[0, "Fcm_diff_avg"] = np.sum(df_pimj.k * df_pimj.pimj)

        # Difference variance
        m_u = np.sum(df_pimj.k * df_pimj.pimj)
        df_feat.loc[0, "Fcm_diff_var"] = np.sum((df_pimj.k - m_u) ** 2.0 * df_pimj.pimj)

        # Difference entropy
        df_feat.loc[0, "Fcm_diff_entr"] = -np.sum(df_pimj.pimj * np.log2(df_pimj.pimj))

        # Sum average
        df_feat.loc[0, "Fcm_sum_avg"] = np.sum(df_pipj.k * df_pipj.pipj)

        # Sum variance
        m_u = np.sum(df_pipj.k * df_pipj.pipj)
        df_feat.loc[0, "Fcm_sum_var"] = np.sum((df_pipj.k - m_u) ** 2.0 * df_pipj.pipj)

        # Sum entropy
        df_feat.loc[0, "Fcm_sum_entr"] = -np.sum(df_pipj.pipj * np.log2(df_pipj.pipj))

        # Angular second moment
        df_feat.loc[0, "Fcm_energy"] = np.sum(df_pij.pij ** 2.0)

        # Contrast
        df_feat.loc[0, "Fcm_contrast"] = np.sum((df_pij.i - df_pij.j) ** 2.0 * df_pij.pij)

        # Dissimilarity
        df_feat.loc[0, "Fcm_dissimilarity"] = np.sum(np.abs(df_pij.i - df_pij.j) * df_pij.pij)

        # Inverse difference
        df_feat.loc[0, "Fcm_inv_diff"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j)))

        # Inverse difference normalised
        df_feat.loc[0, "Fcm_inv_diff_norm"] = np.sum(df_pij.pij / (1.0 + np.abs(df_pij.i - df_pij.j) / n_g))

        # Inverse difference moment
        df_feat.loc[0, "Fcm_inv_diff_mom"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j) ** 2.0))

        # Inverse difference moment normalised
        df_feat.loc[0, "Fcm_inv_diff_mom_norm"] = np.sum(df_pij.pij / (1.0 + (df_pij.i - df_pij.j)
                                                  ** 2.0 / n_g ** 2.0))

        # Inverse variance
        df_sel = df_pij[df_pij.i != df_pij.j]
        df_feat.loc[0, "Fcm_inv_var"] = np.sum(df_sel.pij / (df_sel.i - df_sel.j) ** 2.0)
        del df_sel

        # Correlation
        mu_marg = np.sum(df_pi.i * df_pi.pi)
        var_marg = np.sum((df_pi.i - mu_marg) ** 2.0 * df_pi.pi)

        if var_marg == 0.0:
            df_feat.loc[0, "Fcm_corr"] = 1.0
        else:
            df_feat.loc[0, "Fcm_corr"] = 1.0 / var_marg * (np.sum(df_pij.i * df_pij.j * df_pij.pij) - mu_marg ** 2.0)

        del mu_marg, var_marg

        # Autocorrelation
        df_feat.loc[0, "Fcm_auto_corr"] = np.sum(df_pij.i * df_pij.j * df_pij.pij)

        # Information correlation 1
        hxy = -np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_1 = -np.sum(df_pij.pij * np.log2(df_pij.pi * df_pij.pj))
        hx = -np.sum(df_pi.pi * np.log2(df_pi.pi))
        if len(df_pij) == 1 or hx == 0.0:
            df_feat.loc[0, "Fcm_info_corr1"] = 1.0
        else:
            df_feat.loc[0, "Fcm_info_corr1"] = (hxy - hxy_1) / hx
        del hxy, hxy_1, hx

        # Information correlation 2 - Note: iteration over combinations of i and j
        hxy = - np.sum(df_pij.pij * np.log2(df_pij.pij))
        hxy_2 = - np.sum(
                        np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)) * \
                        np.log2(np.tile(df_pi.pi, len(df_pj)) * np.repeat(df_pj.pj, len(df_pi)))
                        )

        if hxy_2 < hxy:
            df_feat.loc[0, "Fcm_info_corr2"] = 0
        else:
            df_feat.loc[0, "Fcm_info_corr2"] = np.sqrt(1 - np.exp(-2.0 * (hxy_2 - hxy)))
        del hxy, hxy_2

        # Cluster tendency
        m_u = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_tend"] = np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 2.0 * df_pij.pij)
        del m_u

        # Cluster shade
        m_u = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_shade"] = np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 3.0 * df_pij.pij)
        del m_u

        # Cluster prominence
        m_u = np.sum(df_pi.i * df_pi.pi)
        df_feat.loc[0, "Fcm_clust_prom"] = np.sum((df_pij.i + df_pij.j - 2 * m_u) ** 4.0 * df_pij.pij)

        del df_pi, df_pj, df_pij, df_pimj, df_pipj, n_g

        # Update names
        # df_feat.columns += self._parse_names()

        return df_feat

    def _parse_names(self) -> str:
        """"Adds additional settings-related identifiers to each feature.
        Not used currently, as the use of different settings for the
        co-occurrence matrix is not supported.

        Returns:
            str: String of the features indetifier.
        """
        parse_str = ""

        # Add distance
        parse_str += "_d" + str(np.round(self.distance, 1))

        # Add spatial method
        if self.spatial_method is not None:
            parse_str += "_" + self.spatial_method

        # Add merge method
        if self.merge_method is not None:
            if self.merge_method == "average":
                parse_str += "_avg"
            if self.merge_method == "slice_merge":
                parse_str += "_s_mrg"
            if self.merge_method == "dir_merge":
                parse_str += "_d_mrg"
            if self.merge_method == "vol_merge":
                parse_str += "_v_mrg"

        return parse_str
