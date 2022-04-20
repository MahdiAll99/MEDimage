#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np


def getAxisLengths(XYZ) -> Union[float, float, float]:
    """Computes AxisLengths.
    
    Args:
        XYZ (ndarray): Array of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume. In mm.

    Returns:
        Union[float, float, float]: Array of three column vectors
        [Major axis lengths, Minor axis lengths, Least axis lengths].

    """
    XYZ = XYZ.copy()

    # Getting the geometric centre of mass
    com_geom = np.sum(XYZ, 0)/np.shape(XYZ)[0]  # [1 X 3] vector

    # Subtracting the centre of mass
    XYZ[:, 0] = XYZ[:, 0] - com_geom[0]
    XYZ[:, 1] = XYZ[:, 1] - com_geom[1]
    XYZ[:, 2] = XYZ[:, 2] - com_geom[2]

    # Getting the covariance matrix
    covMat = np.cov(XYZ, rowvar=False)

    # Getting the eigenvalues
    eigVal, _ = np.linalg.eig(covMat)
    eigVal = np.sort(eigVal)

    major = eigVal[2]
    minor = eigVal[1]
    least = eigVal[0]

    return major, minor, least
