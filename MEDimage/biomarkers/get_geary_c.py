#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def get_geary_c(vol, res) -> float:
    """Computes Geary'C measure (Assesses intensity differences between voxels).
    
    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.

    Returns:
        float: computed value of Geary'C measure.

    """
    vol = vol.copy()
    res = res.copy()

    # Find the location(s) of all non NaNs voxels
    I, J, K = np.nonzero(~np.isnan(vol))
    n_vox = np.size(I)

    # Get the mean
    u = np.mean(vol[~np.isnan(vol[:])])
    vol_m_mean_s = np.power((vol.copy() - u), 2)  # (Xgl,i - u).^2
    
    # Sum of (Xgl,i - u).^2 over all i
    sum_s = np.sum(vol_m_mean_s[~np.isnan(vol_m_mean_s[:])])

    # Get a meshgrid first
    x = res[0]*((np.arange(1, np.shape(vol)[0]+1))-0.5)
    y = res[1]*((np.arange(1, np.shape(vol)[1]+1))-0.5)
    z = res[2]*((np.arange(1, np.shape(vol)[2]+1))-0.5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    temp = 0
    sum_w = 0

    for i in range(1, n_vox+1):
        # Distance mesh
        temp_x = X - X[I[i-1], J[i-1], K[i-1]]
        temp_y = Y - Y[I[i-1], J[i-1], K[i-1]]
        temp_z = Z - Z[I[i-1], J[i-1], K[i-1]]

        # meshgrid of weigths
        temp_dist_mesh = 1/np.sqrt(temp_x**2 + temp_y**2 + temp_z**2)

        # Removing NaNs
        temp_dist_mesh[np.isnan(vol)] = np.NaN
        temp_dist_mesh[I[i-1], J[i-1], K[i-1]] = np.NaN

        # Running sum of weights
        w_sum = np.sum(temp_dist_mesh[~np.isnan(temp_dist_mesh[:])])
        sum_w = sum_w + w_sum

        # Inside sum calculation
        val = vol[I[i-1], J[i-1], K[i-1]].copy()  # Xgl,i
        # wij.*(Xgl,i - Xgl,j).^2
        temp_vol = temp_dist_mesh*(vol - val)**2

        # Removing i voxel to be sure;
        temp_vol[I[i-1], J[i-1], K[i-1]] = np.NaN

        # Sum of wij.*(Xgl,i - Xgl,j).^2 over all j
        sum_val = np.sum(temp_vol[~np.isnan(temp_vol[:])])

        # Running sum of (sum of wij.*(Xgl,i - Xgl,j).^2 over all j) over all i
        temp = temp + sum_val

    geary_c = temp * (n_vox-1) / sum_s / (2*sum_w)

    return geary_c
