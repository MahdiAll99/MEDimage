#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List

import numpy as np


def find_i_x(levels: np.ndarray,
             fract_vol: np.ndarray,
             x: float) -> np.ndarray:
    """Computes intensity at volume fraction.

    Args:
        levels (ndarray): COMPLETE INTEGER grey-levels.
        fract_vol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of minimum discretised intensity present in at most :math:`x` % of the volume.
    
    """
    ind = np.where(fract_vol <= x/100)[0][0]
    ix = levels[ind]

    return ix
    
def find_v_x(fract_int: np.ndarray,
             fract_vol: np.ndarray,
             x: float) -> np.ndarray:
    """Computes volume at intensity fraction.

    Args:
        fract_int (ndarray): Intensity fraction.
        fract_vol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of largest volume fraction ``fract_vol`` that has an
        intensity fraction ``fract_int`` of at least :math:`x` %.

    """
    ind = np.where(fract_int >= x/100)[0][0]
    vx = fract_vol[ind]

    return vx

def get_glcm_cross_diag_prob(p_ij: np.ndarray) -> np.ndarray:
    """Computes cross diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighboring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the cross diagonal probability.
     
    """
    n_g = np.size(p_ij, 0)
    val_k = np.arange(2, 2*n_g + 100*np.finfo(float).eps)
    n_k = np.size(val_k)
    p_iplusj = np.zeros(n_k)

    for iteration_k in range(0, n_k):
        k = val_k[iteration_k]
        p = 0
        for i in range(0, n_g):
            for j in range(0, n_g):
                if (k - (i+j+2)) == 0:
                    p += p_ij[i, j]

        p_iplusj[iteration_k] = p

    return p_iplusj

def get_glcm_diag_prob(p_ij: np.ndarray) -> np.ndarray:
    """Computes diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighboring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the diagonal probability.
    
    """

    n_g = np.size(p_ij, 0)
    val_k = np.arange(0, n_g)
    n_k = np.size(val_k)
    p_iminusj = np.zeros(n_k)

    for iteration_k in range(0, n_k):
        k = val_k[iteration_k]
        p = 0
        for i in range(0, n_g):
            for j in range(0, n_g):
                if (k - abs(i-j)) == 0:
                    p += p_ij[i, j]

        p_iminusj[iteration_k] = p

    return p_iminusj

def get_loc_peak(img_obj: np.ndarray,
                 roi_obj: np.ndarray,
                 res: np.ndarray) -> float:
    """Computes Local intensity peak.

    Note:
        This works only in 3D for now.

    Args:
        img_obj (ndarray): Continuos image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            xyz resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the local intensity peak.
    
    """
    # INITIALIZATION
    # About 6.2 mm, as defined in document
    dist_thresh = (3/(4*math.pi))**(1/3)*10

    # Insert -inf outside ROI
    temp = img_obj.copy()
    img_obj = img_obj.copy()
    img_obj[roi_obj == 0] = -np.inf

    # Find the location(s) of the maximal voxel
    max_val = np.max(img_obj)
    I, J, K = np.nonzero(img_obj == max_val)
    n_max = np.size(I)

    # Reconverting to full object without -Inf
    img_obj = temp

    # Get a meshgrid first
    x = res[0]*(np.arange(img_obj.shape[1])+0.5)
    y = res[1]*(np.arange(img_obj.shape[0])+0.5)
    z = res[2]*(np.arange(img_obj.shape[2])+0.5)
    X, Y, Z = np.meshgrid(x, y, z)  # In mm

    # Calculate the local peak
    max_val = -np.inf

    for n in range(n_max):
        temp_x = X - X[I[n], J[n], K[n]]
        temp_y = Y - Y[I[n], J[n], K[n]]
        temp_z = Z - Z[I[n], J[n], K[n]]
        temp_dist_mesh = (np.sqrt(np.power(temp_x, 2) + 
                                np.power(temp_y, 2) +
                                np.power(temp_z, 2)))
        val = img_obj[temp_dist_mesh <= dist_thresh]
        val[np.isnan(val)] = []

        if np.size(val) == 0:
            temp_local_peak = img_obj[I[n], J[n], K[n]]
        else:
            temp_local_peak = np.mean(val)
        if temp_local_peak > max_val:
            max_val = temp_local_peak

    local_peak = max_val

    return local_peak

def get_glob_peak(img_obj: np.ndarray,
                  roi_obj: np.ndarray,
                  res: np.ndarray) -> float:
    """Computes Global intensity peak.

    Note:
        This works only in 3D for now.

    Args:
        img_obj (ndarray): Continuos image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            xyz resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the global intensity peak.

    """
    # INITIALIZATION
    # About 6.2 mm, as defined in document
    dist_thresh = (3/(4*math.pi))**(1/3)*10

    # Find the location(s) of all voxels within the ROI
    indices = np.nonzero(np.reshape(roi_obj, np.size(roi_obj), order='F') == 1)[0]
    I, J, K = np.unravel_index(indices, np.shape(img_obj), order='F')
    n_max = np.size(I)

    # Get a meshgrid first
    x = res[0]*(np.arange(img_obj.shape[1])+0.5)
    y = res[1]*(np.arange(img_obj.shape[0])+0.5)
    z = res[2]*(np.arange(img_obj.shape[2])+0.5)
    X, Y, Z = np.meshgrid(x, y, z)  # In mm

    # Calculate the local peak
    max_val = -np.inf

    for n in range(n_max):
        temp_x = X - X[I[n], J[n], K[n]]
        temp_y = Y - Y[I[n], J[n], K[n]]
        temp_z = Z - Z[I[n], J[n], K[n]]
        temp_dist_mesh = (np.sqrt(np.power(temp_x, 2) + 
                                np.power(temp_y, 2) +
                                np.power(temp_z, 2)))
        val = img_obj[temp_dist_mesh <= dist_thresh]
        val[np.isnan(val)] = []

        if np.size(val) == 0:
            temp_local_peak = img_obj[I[n], J[n], K[n]]
        else:
            temp_local_peak = np.mean(val)
        if temp_local_peak > max_val:
            max_val = temp_local_peak

    global_peak = max_val

    return global_peak
    