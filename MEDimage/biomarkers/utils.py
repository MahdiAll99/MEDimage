#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Tuple, Union

import numpy as np
from skimage.measure import marching_cubes


def findIX(levels, fractVol, x) -> np.ndarray:
    """Computes intensity at volume fraction.

    Args:
        levels (ndarray): COMPLETE INTEGER grey-levels.
        fractVol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of minimum discretised intensity present 
            in at most `x`% of the volume.
    
    """
    ind = np.where(fractVol <= x/100)[0][0]
    Ix = levels[ind]

    return Ix
    
def findVX(fractInt, fractVol, x) -> np.ndarray:
    """Computes volume at intensity fraction.

    Args:
        fractInt (ndarray): Intensity fraction.
        fractVol (ndarray): Fractional volume.
        x (float): Fraction percentage, between 0 and 100.

    Returns:
        ndarray: Array of largest volume fraction `fractVol` that has an 
            intensity fraction `fractInt` of at least `x`%.

    """
    ind = np.where(fractInt >= x/100)[0][0]
    Vx = fractVol[ind]

    return Vx

def getAreaDensApprox(a, b, c, n) -> float:
    """Computes area density - minimum volume enclosing ellipsoid
    
    Args:
        a (float): Major semi-axis length.
        b (float): Minor semi-axis length.
        c (float): Least semi-axis length.
        n (int): Number of iterations.

    Returns:
        float: Area density - minimum volume enclosing ellipsoid.

    """
    alpha = np.sqrt(1 - b**2/a**2)
    beta = np.sqrt(1 - c**2/a**2)
    AB = alpha * beta
    point = (alpha**2+beta**2) / (2*AB)
    Aell = 0

    for v in range(0, n+1):
        coef = [0]*v + [1]
        legen = np.polynomial.legendre.legval(x=point, c=coef)
        Aell = Aell + AB**v / (1-4*v**2) * legen

    Aell = Aell * 4 * np.pi * a * b

    return Aell

def getAxisLengths(XYZ) -> Tuple[float, float, float]:
    """Computes AxisLengths.
    
    Args:
        XYZ (ndarray): Array of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume. In mm.

    Returns:
        Tuple[float, float, float]: Array of three column vectors
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

def getGLCMCrossDiagProb(p_ij) -> np.ndarray:
    """Computes cross diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighboring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the cross diagonal probability.
     
    """
    Ng = np.size(p_ij, 0)
    valK = np.arange(2, 2*Ng + 100*np.finfo(float).eps)
    nK = np.size(valK)
    p_iplusj = np.zeros(nK)

    for iterationK in range(0, nK):
        k = valK[iterationK]
        p = 0
        for i in range(0, Ng):
            for j in range(0, Ng):
                if (k - (i+j+2)) == 0:
                    p += p_ij[i, j]

        p_iplusj[iterationK] = p

    return p_iplusj

def getGLCMDiagProb(p_ij) -> np.ndarray:
    """Computes diagonal probabilities.

    Args:
        p_ij (ndarray): Joint probability of grey levels 
            i and j occurring in neighboring voxels. (Elements
            of the  probability distribution for grey level 
            co-occurrences).

    Returns:
        ndarray: Array of the diagonal probability.
    
    """

    Ng = np.size(p_ij, 0)
    valK = np.arange(0, Ng)
    nK = np.size(valK)
    p_iminusj = np.zeros(nK)

    for iterationK in range(0, nK):
        k = valK[iterationK]
        p = 0
        for i in range(0, Ng):
            for j in range(0, Ng):
                if (k - abs(i-j)) == 0:
                    p += p_ij[i, j]

        p_iminusj[iterationK] = p

    return p_iminusj

def getCOM(Xgl_int, Xgl_morph, XYZ_int, XYZ_morph) -> Union[float, np.ndarray]:
    """Calculates center of mass shift (in mm, since resolution is in mm).

    Note: 
        Row positions of "Xgl" and "XYZ" must correspond for each point.
    
    Args:
        Xgl_int (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the intensity mask).
        Xgl_morph (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the morphological mask).
        XYZ_int (ndarray): [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI intensity mesh)
        XYZ_morph (ndarray): [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI morphological mesh)

    Returns:
        Union[float, np.ndarray]: The ROI volume centre of mass.

    """

    # Getting the geometric centre of mass
    Nv = np.size(Xgl_morph)

    com_geom = np.sum(XYZ_morph, 0)/Nv  # [1 X 3] vector

    # Getting the density centre of mass
    XYZ_int[:, 0] = Xgl_int*XYZ_int[:, 0]
    XYZ_int[:, 1] = Xgl_int*XYZ_int[:, 1]
    XYZ_int[:, 2] = Xgl_int*XYZ_int[:, 2]
    com_gl = np.sum(XYZ_int, 0)/np.sum(Xgl_int, 0)  # [1 X 3] vector

    # Calculating the shift
    com = np.linalg.norm(com_geom - com_gl)

    return com

def getLocPeak(imgObj, roiObj, res) -> float:
    """Computes Local intensity peak.

    Note:
        This works only in 3D for now.

    Args:
        imgObj (ndarray): Continuos image intensity distribution, with no NaNs
            outside the ROI.
        roiObj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the local intensity peak.
    
    """
    # INITIALIZATION
    # About 6.2 mm, as defined in document
    distThresh = (3/(4*math.pi))**(1/3)*10

    # Insert -inf outside ROI
    temp = imgObj.copy()
    imgObj = imgObj.copy()
    imgObj[roiObj == 0] = -np.inf

    # Find the location(s) of the maximal voxel
    maxVal = np.max(imgObj)
    I, J, K = np.nonzero(imgObj == maxVal)
    nMax = np.size(I)

    # Reconverting to full object without -Inf
    imgObj = temp

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

    localPeak = maxVal

    return localPeak

def getMesh(mask, res) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mesh.

    Note:
      Make sure the `mask` is padded with a layer of 0's in all
      dimensions to reduce potential isosurface computation errors.

    Args:
        mask (ndarray): Contains only 0's and 1's.
        res (ndarray or List): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Array of the [X,Y,Z] positions of the ROI.
            - Array of the spatial coordinates for `mask` unique mesh vertices.
            - Array of triangular faces via referencing vertex indices from vertices.
    """
    # Getting the grid of X,Y,Z positions, where the coordinate reference
    # system (0,0,0) is located at the upper left corner of the first voxel
    # (-0.5: half a voxel distance). For the whole volume defining the mask,
    # no matter if it is a 1 or a 0.
    mask = mask.copy()
    res = res.copy()

    x = res[0]*((np.arange(1, np.shape(mask)[0]+1))-0.5)
    y = res[1]*((np.arange(1, np.shape(mask)[1]+1))-0.5)
    z = res[2]*((np.arange(1, np.shape(mask)[2]+1))-0.5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Getting the isosurface of the mask
    vertices, faces, _, _ = marching_cubes(volume=mask, level=0.5, spacing=res)

    # Getting the X,Y,Z positions of the ROI (i.e. 1's) of the mask
    X = np.reshape(X, (np.size(X), 1), order='F')
    Y = np.reshape(Y, (np.size(Y), 1), order='F')
    Z = np.reshape(Z, (np.size(Z), 1), order='F')

    XYZ = np.concatenate((X, Y, Z), axis=1)
    XYZ = XYZ[np.where(np.reshape(mask, np.size(mask), order='F') == 1)[0], :]

    return XYZ, faces, vertices

def getGlobPeak(imgObj, roiObj, res) -> float:
    """Computes Global intensity peak.

    Note:
        This works only in 3D for now.

    Args:
        imgObj (ndarray): Continuos image intensity distribution, with no NaNs
            outside the ROI.
        roiObj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
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
    