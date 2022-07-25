#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Tuple, Union

import numpy as np
from skimage.measure import marching_cubes


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

def get_area_dens_approx(a: float,
                         b: float,
                         c: float,
                         n: float) -> float:
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
    ab = alpha * beta
    point = (alpha**2+beta**2) / (2*ab)
    a_ell = 0

    for v in range(0, n+1):
        coef = [0]*v + [1]
        legen = np.polynomial.legendre.legval(x=point, c=coef)
        a_ell = a_ell + ab**v / (1-4*v**2) * legen

    a_ell = a_ell * 4 * np.pi * a * b

    return a_ell

def get_axis_lengths(xyz: np.ndarray) -> Tuple[float, float, float]:
    """Computes AxisLengths.
    
    Args:
        xyz (ndarray): Array of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume. In mm.

    Returns:
        Tuple[float, float, float]: Array of three column vectors 
            [Major axis lengths, Minor axis lengths, Least axis lengths].

    """
    xyz = xyz.copy()

    # Getting the geometric centre of mass
    com_geom = np.sum(xyz, 0)/np.shape(xyz)[0]  # [1 X 3] vector

    # Subtracting the centre of mass
    xyz[:, 0] = xyz[:, 0] - com_geom[0]
    xyz[:, 1] = xyz[:, 1] - com_geom[1]
    xyz[:, 2] = xyz[:, 2] - com_geom[2]

    # Getting the covariance matrix
    cov_mat = np.cov(xyz, rowvar=False)

    # Getting the eigenvalues
    eig_val, _ = np.linalg.eig(cov_mat)
    eig_val = np.sort(eig_val)

    major = eig_val[2]
    minor = eig_val[1]
    least = eig_val[0]

    return major, minor, least

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

def get_com(xgl_int: np.ndarray,
            xgl_morph: np.ndarray,
            xyz_int: np.ndarray,
            xyz_morph: np.ndarray) -> Union[float,
                                            np.ndarray]:
    """Calculates center of mass shift (in mm, since resolution is in mm).

    Note: 
        Row positions of "x_gl" and "xyz" must correspond for each point.
    
    Args:
        xgl_int (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the intensity mask).
        xgl_morph (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the morphological mask).
        xyz_int (ndarray): [n_points X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI intensity mesh)
        xyz_morph (ndarray): [n_points X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI morphological mesh)

    Returns:
        Union[float, np.ndarray]: The ROI volume centre of mass.

    """

    # Getting the geometric centre of mass
    n_v = np.size(xgl_morph)

    com_geom = np.sum(xyz_morph, 0)/n_v  # [1 X 3] vector

    # Getting the density centre of mass
    xyz_int[:, 0] = xgl_int*xyz_int[:, 0]
    xyz_int[:, 1] = xgl_int*xyz_int[:, 1]
    xyz_int[:, 2] = xgl_int*xyz_int[:, 2]
    com_gl = np.sum(xyz_int, 0)/np.sum(xgl_int, 0)  # [1 X 3] vector

    # Calculating the shift
    com = np.linalg.norm(com_geom - com_gl)

    return com

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

def get_mesh(mask: np.ndarray,
             res: Union[np.ndarray, List]) -> Tuple[np.ndarray,
                                                    np.ndarray,
                                                    np.ndarray]:
    """Compute Mesh.

    Note:
      Make sure the `mask` is padded with a layer of 0's in all
      dimensions to reduce potential isosurface computation errors.

    Args:
        mask (ndarray): Contains only 0's and 1's.
        res (ndarray or List): [a,b,c] vector specifying the resolution of the volume in mm.
            xyz resolution (world), or JIK resolution (intrinsic matlab).

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

    xyz = np.concatenate((X, Y, Z), axis=1)
    xyz = xyz[np.where(np.reshape(mask, np.size(mask), order='F') == 1)[0], :]

    return xyz, faces, vertices

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
    