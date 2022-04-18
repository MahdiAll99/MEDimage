#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getMoranI(vol, res):
    """Compute MoranI.
    -------------------------------------------------------------------------
    AUTHOR(S): MEDomicsLab consortium
    -------------------------------------------------------------------------
    STATEMENT:
    This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
    a package providing MATLAB programming tools for radiomics analysis.
     --> Copyright (C) MEDomicsLab consortium.

    This package is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This package is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this package.  If not, see <http://www.gnu.org/licenses/>.
    -------------------------------------------------------------------------
    """

    vol = vol.copy()
    res = res.copy()

    # Find the location(s) of all non NaNs voxels
    I, J, K = np.nonzero(~np.isnan(vol))
    nVox = np.size(I)

    # Get the mean
    u = np.mean(vol[~np.isnan(vol[:])])
    volMmean = vol.copy() - u  # (Xgl,i - u)
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
        # Removing NaNs
        tempVol = volMmean.copy()
        tempVol[I[i-1], J[i-1], K[i-1]] = np.NaN
        tempVol = tempDistMesh * tempVol  # (wij .* (Xgl,j - u))
        # Summing (wij .* (Xgl,j - u)) over all j
        sumVal = np.sum(tempVol[~np.isnan(tempVol[:])])
        # Running sum of (Xgl,i - u)*(wij .* (Xgl,j - u)) over all i
        temp = temp + volMmean[I[i-1], J[i-1], K[i-1]] * sumVal

    moranI = temp*nVox/sumS/sumW

    return moranI
