#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np


def getGlobPeak(imgObj, roiObj, res) -> float:
    """Computes Global intensity peak.

    Note:
        This works only in 3D for now.

    Args:
        imgObj (ndarray): Continous image intentisity distribution, with no NaNs
            outside the ROI.
        roiObj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the global intensity peak.

    """
    # INITIALIZATION
    # About 6.2 mm, as defined in document
    distThresh = (3/(4*math.pi))**(1/3)*10

    # Find the location(s) of all voxels within the ROI
    indices = np.nonzero(np.reshape(roiObj, np.size(roiObj), order='F') == 1)[0]
    I, J, K = np.unravel_index(indices, np.shape(imgObj), order='F')
    nMax = np.size(I)

    # Get a meshgrid first
    x = res[0]*(np.arange(imgObj.shape[1])+0.5)
    y = res[1]*(np.arange(imgObj.shape[0])+0.5)
    z = res[2]*(np.arange(imgObj.shape[2])+0.5)
    X, Y, Z = np.meshgrid(x, y, z)  # In mm

    # Calculate the local peak
    maxVal = -np.inf

    for n in range(nMax):
        tempX = X - X[I[n], J[n], K[n]]
        tempY = Y - Y[I[n], J[n], K[n]]
        tempZ = Z - Z[I[n], J[n], K[n]]
        tempDistMesh = (np.sqrt(np.power(tempX, 2) + 
                                np.power(tempY, 2) +
                                np.power(tempZ, 2)))
        val = imgObj[tempDistMesh <= distThresh]
        val[np.isnan(val)] = []

        if np.size(val) == 0:
            tempLocalPeak = imgObj[I[n], J[n], K[n]]
        else:
            tempLocalPeak = np.mean(val)
        if tempLocalPeak > maxVal:
            maxVal = tempLocalPeak

    globalPeak = maxVal

    return globalPeak
