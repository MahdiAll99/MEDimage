#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def getGearyC(vol, res) -> float:
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
    nVox = np.size(I)

    # Get the mean
    u = np.mean(vol[~np.isnan(vol[:])])
    volMmeanS = np.power((vol.copy() - u), 2)  # (Xgl,i - u).^2
    
    # Sum of (Xgl,i - u).^2 over all i
    sumS = np.sum(volMmeanS[~np.isnan(volMmeanS[:])])

    # Get a meshgrid first
    x = res[0]*((np.arange(1, np.shape(vol)[0]+1))-0.5)
    y = res[1]*((np.arange(1, np.shape(vol)[1]+1))-0.5)
    z = res[2]*((np.arange(1, np.shape(vol)[2]+1))-0.5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    temp = 0
    sumW = 0

    for i in range(1, nVox+1):
        # Distance mesh
        tempX = X - X[I[i-1], J[i-1], K[i-1]]
        tempY = Y - Y[I[i-1], J[i-1], K[i-1]]
        tempZ = Z - Z[I[i-1], J[i-1], K[i-1]]

        # meshgrid of weigths
        tempDistMesh = 1/np.sqrt(tempX**2 + tempY**2 + tempZ**2)

        # Removing NaNs
        tempDistMesh[np.isnan(vol)] = np.NaN
        tempDistMesh[I[i-1], J[i-1], K[i-1]] = np.NaN

        # Running sum of weights
        wsum = np.sum(tempDistMesh[~np.isnan(tempDistMesh[:])])
        sumW = sumW + wsum

        # Inside sum calculation
        val = vol[I[i-1], J[i-1], K[i-1]].copy()  # Xgl,i
        # wij.*(Xgl,i - Xgl,j).^2
        tempVol = tempDistMesh*(vol - val)**2

        # Removing i voxel to be sure;
        tempVol[I[i-1], J[i-1], K[i-1]] = np.NaN

        # Sum of wij.*(Xgl,i - Xgl,j).^2 over all j
        sumVal = np.sum(tempVol[~np.isnan(tempVol[:])])

        # Running sum of (sum of wij.*(Xgl,i - Xgl,j).^2 over all j) over all i
        temp = temp + sumVal

    gearyC = temp * (nVox-1) / sumS / (2*sumW)

    return gearyC
