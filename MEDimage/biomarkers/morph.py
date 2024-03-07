#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.spatial as sc
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes

from ..biomarkers.get_oriented_bound_box import rot_matrix, sig_proc_find_peaks


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

def min_oriented_bound_box(pos_mat: np.ndarray) -> np.ndarray:
    """Computes the minimum bounding box of an arbitrary solid: an iterative approach.
    This feature refers to "Volume density (oriented minimum bounding box)" (ID = ZH1A)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        pos_mat (ndarray): matrix position

    Returns:
        ndarray: return bounding box dimensions
    """

    ##########################
    # Internal functions
    ##########################

    def calc_rot_aabb_surface(theta: float,
                              hull_mat: np.ndarray) -> np.ndarray:
        """Function to calculate surface of the axis-aligned bounding box of a rotated 2D contour

        Args:
            theta (float): angle in radian
            hull_mat (nddarray): convex hull matrix

        Returns:
            ndarray: the surface of the axis-aligned bounding box of a rotated 2D contour
        """

        # Create rotation matrix and rotate over theta
        rot_mat = rot_matrix(theta=theta, dim=2)
        rot_hull = np.dot(rot_mat, hull_mat)

        # Calculate bounding box surface of the rotated contour
        rot_aabb_dims = np.max(rot_hull, axis=1) - np.min(rot_hull, axis=1)
        rot_aabb_area = np.product(rot_aabb_dims)

        return rot_aabb_area

    def approx_min_theta(hull_mat: np.ndarray,
                         theta_sel: float,
                         res: float,
                         max_rep: int=5) -> np.ndarray:
        """Iterative approximator for finding angle theta that minimises surface area

        Args:
            hull_mat (ndarray): convex hull matrix
            theta_sel (float): angle in radian
            res (float): value in radian
            max_rep (int, optional): maximum repetition. Defaults to 5.

        Returns:
            ndarray: the angle theta that minimises surfae area
        """

        for i in np.arange(0, max_rep):

            # Select new thetas in vicinity of
            theta = np.array([theta_sel-res, theta_sel-0.5*res,
                              theta_sel, theta_sel+0.5*res, theta_sel+res])

            # Calculate projection areas for current angles theta
            rot_area = np.array(
                list(map(lambda x: calc_rot_aabb_surface(theta=x, hull_mat=hull_mat), theta)))

            # Find global minimum and corresponding angle theta_sel
            theta_sel = theta[np.argmin(rot_area)]

            # Shrink resolution and iterate
            res = res / 2.0

        return theta_sel

    def rotate_minimal_projection(input_pos: float,
                                  rot_axis: int,
                                  n_minima: int=3,
                                  res_init: float=5.0):
        """Function to that rotates input_pos to find the rotation that
        minimises the projection of input_pos on the
        plane normal to the rot_axis

        Args:
            input_pos (float): input position value
            rot_axis (int): rotation axis value
            n_minima (int, optional): _description_. Defaults to 3.
            res_init (float, optional): _description_. Defaults to 5.0.

        Returns:
            _type_: _description_
        """


        # Find axis aligned bounding box of the point set
        aabb_max = np.max(input_pos, axis=0)
        aabb_min = np.min(input_pos, axis=0)

        # Center the point set at the AABB center
        output_pos = input_pos - 0.5 * (aabb_min + aabb_max)

        # Project model to plane
        proj_pos = copy.deepcopy(output_pos)
        proj_pos = np.delete(proj_pos, [rot_axis], axis=1)

        # Calculate 2D convex hull of the model projection in plane
        if np.shape(proj_pos)[0] >= 10:
            hull_2d = ConvexHull(points=proj_pos)
            hull_mat = proj_pos[hull_2d.vertices, :]
            del hull_2d, proj_pos
        else:
            hull_mat = proj_pos
            del proj_pos

        # Transpose hull_mat so that the array is (ndim, npoints) instead of (npoints, ndim)
        hull_mat = np.transpose(hull_mat)

        # Calculate bounding box surface of a series of rotated contours
        # Note we can program a min-search algorithm here as well

        # Calculate initial surfaces
        theta_init = np.arange(start=0.0, stop=90.0 +
                               res_init, step=res_init) * np.pi / 180.0
        rot_area = np.array(
            list(map(lambda x: calc_rot_aabb_surface(theta=x, hull_mat=hull_mat), theta_init)))

        # Find local minima
        df_min = sig_proc_find_peaks(x=rot_area, ddir="neg")

        # Check if any minimum was generated
        if len(df_min) > 0:
            # Investigate up to n_minima number of local minima, starting with the global minimum
            df_min = df_min.sort_values(by="val", ascending=True)

            # Determine max number of minima evaluated
            max_iter = np.min([n_minima, len(df_min)])

            # Initialise place holder array
            theta_min = np.zeros(max_iter)

            # Iterate over local minima
            for k in np.arange(0, max_iter):

                # Find initial angle corresponding to i-th minimum
                sel_ind = df_min.ind.values[k]
                theta_curr = theta_init[sel_ind]

                # Zoom in to improve the approximation of theta
                theta_min[k] = approx_min_theta(
                    hull_mat=hull_mat, theta_sel=theta_curr, res=res_init*np.pi/180.0)

            # Calculate surface areas corresponding to theta_min and theta that
            # minimises the surface
            rot_area = np.array(
                list(map(lambda x: calc_rot_aabb_surface(theta=x, hull_mat=hull_mat), theta_min)))
            theta_sel = theta_min[np.argmin(rot_area)]

        else:
            theta_sel = theta_init[0]

        # Rotate original point along the angle that minimises the projected AABB area
        output_pos = np.transpose(output_pos)
        rot_mat = rot_matrix(theta=theta_sel, dim=3, rot_axis=rot_axis)
        output_pos = np.dot(rot_mat, output_pos)

        # Rotate output_pos back to (npoints, ndim)
        output_pos = np.transpose(output_pos)

        return output_pos

    ##########################
    # Main function
    ##########################

    rot_df = pd.DataFrame({"rot_axis_0":  np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
                           "rot_axis_1":  np.array([1, 2, 1, 2, 0, 2, 0, 2, 0, 1, 0, 1]),
                           "rot_axis_2":  np.array([2, 1, 0, 0, 2, 0, 1, 1, 1, 0, 2, 2]),
                           "aabb_axis_0": np.zeros(12),
                           "aabb_axis_1": np.zeros(12),
                           "aabb_axis_2": np.zeros(12),
                           "vol":         np.zeros(12)})

    # Rotate over different sequences
    for i in np.arange(0, len(rot_df)):
        # Create a local copy
        work_pos = copy.deepcopy(pos_mat)

        # Rotate over sequence of rotation axes
        work_pos = rotate_minimal_projection(
            input_pos=work_pos, rot_axis=rot_df.rot_axis_0[i])
        work_pos = rotate_minimal_projection(
            input_pos=work_pos, rot_axis=rot_df.rot_axis_1[i])
        work_pos = rotate_minimal_projection(
            input_pos=work_pos, rot_axis=rot_df.rot_axis_2[i])

        # Determine resultant minimum bounding box
        aabb_dims = np.max(work_pos, axis=0) - np.min(work_pos, axis=0)
        rot_df.loc[i, "aabb_axis_0"] = aabb_dims[0]
        rot_df.loc[i, "aabb_axis_1"] = aabb_dims[1]
        rot_df.loc[i, "aabb_axis_2"] = aabb_dims[2]
        rot_df.loc[i, "vol"] = np.product(aabb_dims)

        del work_pos, aabb_dims

    # Find minimal volume of all rotations and return bounding box dimensions
    idxmin = rot_df.vol.idxmin()
    sel_row = rot_df.loc[idxmin]
    ombb_dims = np.array(
        [sel_row.aabb_axis_0, sel_row.aabb_axis_1, sel_row.aabb_axis_2])

    return ombb_dims

def get_moran_i(vol: np.ndarray,
                res: List[float]) -> float:
    """Computes Moran's Index.
    This feature refers to "Moran’s I index" (ID = N365)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        res (List[float]): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world) or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of Moran's Index.

    """
    vol = vol.copy()
    res = res.copy()

    # Find the location(s) of all non NaNs voxels
    I, J, K = np.nonzero(~np.isnan(vol))
    n_vox = np.size(I)

    # Get the mean
    u = np.mean(vol[~np.isnan(vol[:])])
    vol_mean = vol.copy() - u  # (x_gl,i - u)
    vol_m_mean_s = np.power((vol.copy() - u), 2)  # (x_gl,i - u).^2
    # Sum of (x_gl,i - u).^2 over all i
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
        temp_dist_mesh = 1 / np.sqrt(temp_x**2 + temp_y**2 + temp_z**2)

        # Removing NaNs
        temp_dist_mesh[np.isnan(vol)] = np.NaN
        temp_dist_mesh[I[i-1], J[i-1], K[i-1]] = np.NaN
        # Running sum of weights
        w_sum = np.sum(temp_dist_mesh[~np.isnan(temp_dist_mesh[:])])
        sum_w = sum_w + w_sum

        # Inside sum calculation
        # Removing NaNs
        temp_vol = vol_mean.copy()
        temp_vol[I[i-1], J[i-1], K[i-1]] = np.NaN
        temp_vol = temp_dist_mesh * temp_vol  # (wij .* (x_gl,j - u))
        # Summing (wij .* (x_gl,j - u)) over all j
        sum_val = np.sum(temp_vol[~np.isnan(temp_vol[:])])
        # Running sum of (x_gl,i - u)*(wij .* (x_gl,j - u)) over all i
        temp = temp + vol_mean[I[i-1], J[i-1], K[i-1]] * sum_val

    moran_i = temp*n_vox/sum_s/sum_w

    return moran_i

def get_mesh_volume(faces: np.ndarray,
                    vertices:np.ndarray) -> float:
    """Computes MeshVolume feature.
    This feature refers to "Volume (mesh)" (ID = RNU0)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        faces (np.ndarray): matrix of three column vectors, defining the [X,Y,Z]
                          positions of the ``faces`` of the isosurface or convex hull of the mask
                          (output from "isosurface.m" or "convhull.m" functions of MATLAB).
                          --> These are more precisely indexes to ``vertices``
        vertices (np.ndarray): matrix of three column vectors, defining the
                             [X,Y,Z] positions of the ``vertices`` of the isosurface of the mask (output
                             from "isosurface.m" function of MATLAB).
                             --> In mm.

    Returns:
        float: Mesh volume
    """
    faces = faces.copy()
    vertices = vertices.copy()

    # Getting vectors for the three vertices
    # (with respect to origin) of each face
    a = vertices[faces[:, 0], :]
    b = vertices[faces[:, 1], :]
    c = vertices[faces[:, 2], :]

    # Calculating volume
    v_cross = np.cross(b, c)
    v_dot = np.sum(a.conj()*v_cross, axis=1)
    volume = np.abs(np.sum(v_dot))/6

    return volume

def get_mesh_area(faces: np.ndarray,
                  vertices: np.ndarray) -> float:
    """Computes the surface area (mesh) feature from the ROI mesh by 
    summing over the triangular face surface areas. 
    This feature refers to "Surface area (mesh)" (ID = C0JK)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        faces (np.ndarray): matrix of three column vectors, defining the [X,Y,Z]
                          positions of the ``faces`` of the isosurface or convex hull of the mask
                          (output from "isosurface.m" or "convhull.m" functions of MATLAB).
                          --> These are more precisely indexes to ``vertices``
        vertices (np.ndarray): matrix of three column vectors,
                             defining the [X,Y,Z]
                             positions of the ``vertices`` of the isosurface of the mask (output
                             from "isosurface.m" function of MATLAB).
                             --> In mm.

    Returns:
        float: Mesh area.
    """

    faces = faces.copy()
    vertices = vertices.copy()

    # Getting two vectors of edges for each face
    a = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    b = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]

    # Calculating the surface area of each face and summing it up all at once.
    c = np.cross(a, b)
    area = 1/2 * np.sum(np.sqrt(np.sum(np.power(c, 2), 1)))

    return area

def get_geary_c(vol: np.ndarray,
                res: np.ndarray) -> float:
    """Computes Geary'C measure (Assesses intensity differences between voxels).
    This feature refers to "Geary's C measure" (ID = NPT7) 
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.

    Returns:
        float: computes value of Geary'C measure.
    """
    vol = vol.copy()
    res = res.copy()

    # Find the location(s) of all non NaNs voxels
    I, J, K = np.nonzero(~np.isnan(vol))
    n_vox = np.size(I)

    # Get the mean
    u = np.mean(vol[~np.isnan(vol[:])])
    vol_m_mean_s = np.power((vol.copy() - u), 2)  # (x_gl,i - u).^2

    # Sum of (x_gl,i - u).^2 over all i
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
        val = vol[I[i-1], J[i-1], K[i-1]].copy()  # x_gl,i
        # wij.*(x_gl,i - x_gl,j).^2
        temp_vol = temp_dist_mesh*(vol - val)**2

        # Removing i voxel to be sure;
        temp_vol[I[i-1], J[i-1], K[i-1]] = np.NaN

        # Sum of wij.*(x_gl,i - x_gl,j).^2 over all j
        sum_val = np.sum(temp_vol[~np.isnan(temp_vol[:])])

        # Running sum of (sum of wij.*(x_gl,i - x_gl,j).^2 over all j) over all i
        temp = temp + sum_val

    geary_c = temp * (n_vox-1) / sum_s / (2*sum_w)

    return geary_c

def min_vol_ellipse(P: np.ndarray,
                    tolerance: np.ndarray) -> Tuple[np.ndarray,
                                                    np.ndarray]:
    """Computes min_vol_ellipse.
    
    Finds the minimum volume enclsing ellipsoid (MVEE) of a set of data
    points stored in matrix P. The following optimization problem is solved:

        minimize $$log(det(A))$$ subject to $$(P_i - c)' * A * (P_i - c) <= 1$$

    in variables A and c, where `P_i` is the `i-th` column of the matrix `P`.
    The solver is based on Khachiyan Algorithm, and the final solution
    is different from the optimal value by the pre-spesified amount of
    `tolerance`.
    
    Note:
        Adapted from MATLAB code of Nima Moshtagh (nima@seas.upenn.edu)
        University of Pennsylvania.

    Args:
        P (ndarray): (d x N) dimnesional matrix containing N points in R^d.
        tolerance (ndarray): error in the solution with respect to the optimal value.
    
    Returns:
        2-element tuple containing
    
        - A: (d x d) matrix of the ellipse equation in the 'center form': \
        $$(x-c)' * A * (x-c) = 1$$ \
        where d is shape of `P` along 0-axis. 
        
        - c: d-dimensional vector as the center of the ellipse. 

    Examples:

        >>>P = rand(5,100)

        >>>[A, c] = :func:`min_vol_ellipse(P, .01)`

        To reduce the computation time, work with the boundary points only:

        >>>K = :func:`convhulln(P)`

        >>>K = :func:`unique(K(:))`

        >>>Q = :func:`P(:,K)`

        >>>[A, c] = :func:`min_vol_ellipse(Q, .01)`
    """

    # Solving the Dual problem
    # data points
    d, N = np.shape(P)
    Q = np.ones((d+1, N))
    Q[:-1, :] = P[:, :]

    # initializations
    err = 1
    u = np.ones(N)/N  # 1st iteration
    new_u = np.zeros(N)

    # Khachiyan Algorithm

    while (err > tolerance):
        diag_u = np.diag(u)
        trans_q = np.transpose(Q)
        X = Q @ diag_u @ trans_q

        # M the diagonal vector of an NxN matrix
        inv_x = np.linalg.inv(X)
        M = np.diag(trans_q @ inv_x @ Q)
        maximum = np.max(M)
        j = np.argmax(M)

        step_size = (maximum - d - 1)/((d+1)*(maximum-1))
        new_u = (1 - step_size)*u.copy()
        new_u[j] = new_u[j] + step_size
        err = np.linalg.norm(new_u - u)
        u = new_u.copy()

    # Computing the Ellipse parameters
    # Finds the ellipse equation in the 'center form':
    # (x-c)' * A * (x-c) = 1
    # It computes a dxd matrix 'A' and a d dimensional vector 'c' as the center
    # of the ellipse.
    U = np.diag(u)

    # the A matrix for the ellipse
    c = P @ u
    c = np.reshape(c, (np.size(c), 1), order='F')  # center of the ellipse

    pup_t = P @ U @ np.transpose(P)
    cct = c @ np.transpose(c)
    a_inv = np.linalg.inv(pup_t - cct)
    A = (1/d) * a_inv

    return A, c

def padding(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Padding the volume and masks.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.

    Returns:
        tuple of 3 ndarray: Volume and masks after padding.
    """

    # PADDING THE VOLUME WITH A LAYER OF NaNs
    # (reduce mesh computation errors of associated mask)
    vol = vol.copy()
    vol = np.pad(vol, pad_width=1, mode="constant", constant_values=np.NaN)
    # PADDING THE MASKS WITH A LAYER OF 0's
    # (reduce mesh computation errors of associated mask)
    mask_int = mask_int.copy()
    mask_int = np.pad(mask_int, pad_width=1, mode="constant", constant_values=0.0)
    mask_morph = mask_morph.copy()
    mask_morph = np.pad(mask_morph, pad_width=1, mode="constant", constant_values=0.0)

    return vol, mask_int, mask_morph

def get_variables(vol: np.ndarray, 
                  mask_int: np.ndarray, 
                  mask_morph: np.ndarray,
                  res: np.ndarray) -> Tuple[np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray, 
                                            np.ndarray]:
    """Compute variables usefull to calculate morphological features.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        tuple of 7 ndarray: Variables usefull to calculate morphological features.
    """
    # GETTING IMPORTANT VARIABLES
    xgl_int = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_int, np.size(mask_int), order='F') == 1)[0]].copy()
    xgl_morph = np.reshape(vol, np.size(vol), order='F')[np.where(
        np.reshape(mask_morph, np.size(mask_morph), order='F') == 1)[0]].copy()
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_int, _, _ = get_mesh(mask_int, res)
    # XYZ refers to [Xc,Yc,Zc] in ref. [1].
    xyz_morph, faces, vertices = get_mesh(mask_morph, res)
    # [X,Y,Z] points of the convex hull.
    # conv_hull Matlab is conv_hull.simplices
    conv_hull = sc.ConvexHull(vertices)

    return xgl_int, xgl_morph, xyz_int, xyz_morph, faces, vertices, conv_hull

def extract_all(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray, 
                intensity_type: str,
                compute_moran_i: bool=False, 
                compute_geary_c: bool=False) -> Dict:
    """Compute Morphological Features.
    This features refer to Morphological family in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.
    
    Note:
        Moran's Index and Geary's C measure takes so much computation time. Please
        use `compute_moran_i` `compute_geary_c` carefully.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        intensity_type (str): Type of intensity to compute. Can be "arbitrary", "definite" or "filtered".
            Will compute all features for "definite" intensity type and all but intergrated intensity for 
            "arbitrary" intensity type.
        compute_moran_i (bool, optional): True to compute Moran's Index.
        compute_geary_c (bool, optional): True to compute Geary's C measure.

    Raises:
        ValueError: If `intensity_type` is not "arbitrary", "definite" or "filtered".
    """
    assert intensity_type in ["arbitrary", "definite", "filtered"], \
        "intensity_type must be 'arbitrary', 'definite' or 'filtered'"
    
    # Initialization of final structure (Dictionary) containing all features.
    morph = {'Fmorph_vol': [],
             'Fmorph_approx_vol': [],
             'Fmorph_area': [],
             'Fmorph_av': [],
             'Fmorph_comp_1': [],
             'Fmorph_comp_2': [],
             'Fmorph_sph_dispr': [],
             'Fmorph_sphericity': [],
             'Fmorph_asphericity': [],
             'Fmorph_com': [],
             'Fmorph_diam': [],
             'Fmorph_pca_major': [],
             'Fmorph_pca_minor': [],
             'Fmorph_pca_least': [],
             'Fmorph_pca_elongation': [],
             'Fmorph_pca_flatness': [],  # until here
             'Fmorph_v_dens_aabb': [],
             'Fmorph_a_dens_aabb': [],
             'Fmorph_v_dens_ombb': [],
             'Fmorph_a_dens_ombb': [],
             'Fmorph_v_dens_aee': [],
             'Fmorph_a_dens_aee': [],
             'Fmorph_v_dens_mvee': [],
             'Fmorph_a_dens_mvee': [],
             'Fmorph_v_dens_conv_hull': [],
             'Fmorph_a_dens_conv_hull': [],
             'Fmorph_integ_int': [],
             'Fmorph_moran_i': [],
             'Fmorph_geary_c': []
            }
    #Initialization
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, xgl_morph, xyz_int, xyz_morph, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)

    # STARTING COMPUTATION
    if intensity_type != "filtered":
        # Volume in mm^3
        volume = get_mesh_volume(faces, vertices)
        morph['Fmorph_vol'] = volume  # Volume

        # Approximate Volume
        morph['Fmorph_approx_vol'] = np.sum(mask_morph[:]) * np.prod(res)

        # Surface area in mm^2
        area = get_mesh_area(faces, vertices)
        morph['Fmorph_area'] = area

        # Surface to volume ratio
        morph['Fmorph_av'] = area / volume

        # Compactness 1
        morph['Fmorph_comp_1'] = volume / ((np.pi**(1/2))*(area**(3/2)))

        # Compactness 2
        morph['Fmorph_comp_2'] = 36*np.pi*(volume**2) / (area**3)

        # Spherical disproportion
        morph['Fmorph_sph_dispr'] = area / (36*np.pi*volume**2)**(1/3)

        # Sphericity
        morph['Fmorph_sphericity'] = ((36*np.pi*volume**2)**(1/3)) / area

        # Asphericity
        morph['Fmorph_asphericity'] = ((area**3) / (36*np.pi*volume**2))**(1/3) - 1

        # Centre of mass shift
        morph['Fmorph_com'] = get_com(xgl_int, xgl_morph, xyz_int, xyz_morph)

        # Maximum 3D diameter
        morph['Fmorph_diam'] = np.max(sc.distance.pdist(conv_hull.points[conv_hull.vertices]))

        # Major axis length
        [major, minor, least] = get_axis_lengths(xyz_morph)
        morph['Fmorph_pca_major'] = 4 * np.sqrt(major)

        # Minor axis length
        morph['Fmorph_pca_minor'] = 4 * np.sqrt(minor)

        # Least axis length
        morph['Fmorph_pca_least'] = 4 * np.sqrt(least)

        # Elongation
        morph['Fmorph_pca_elongation'] = np.sqrt(minor / major)

        # Flatness
        morph['Fmorph_pca_flatness'] = np.sqrt(least / major)

        # Volume density - axis-aligned bounding box
        xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
        yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
        zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
        v_aabb = xc_aabb * yc_aabb * zc_aabb
        morph['Fmorph_v_dens_aabb'] = volume / v_aabb

        # Area density - axis-aligned bounding box
        a_aabb = 2*xc_aabb*yc_aabb + 2*xc_aabb*zc_aabb + 2*yc_aabb*zc_aabb
        morph['Fmorph_a_dens_aabb'] = area / a_aabb

        # Volume density - oriented minimum bounding box
        # Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
        # Determination of the minimum bounding box of an
        # arbitrary solid: an iterative approach.
        # Comp Struc 79 (2001) 1433-1449
        bound_box_dims = min_oriented_bound_box(vertices)
        vol_bb = np.prod(bound_box_dims)
        morph['Fmorph_v_dens_ombb'] = volume / vol_bb

        # Area density - oriented minimum bounding box
        a_ombb = 2 * (bound_box_dims[0]*bound_box_dims[1] +
                        bound_box_dims[0]*bound_box_dims[2] +
                        bound_box_dims[1]*bound_box_dims[2])
        morph['Fmorph_a_dens_ombb'] = area / a_ombb

        # Volume density - approximate enclosing ellipsoid
        a = 2*np.sqrt(major)
        b = 2*np.sqrt(minor)
        c = 2*np.sqrt(least)
        v_aee = (4*np.pi*a*b*c) / 3
        morph['Fmorph_v_dens_aee'] = volume / v_aee

        # Area density - approximate enclosing ellipsoid
        a_aee = get_area_dens_approx(a, b, c, 20)
        morph['Fmorph_a_dens_aee'] = area / a_aee

        # Volume density - minimum volume enclosing ellipsoid
        # (Rotate the volume first??)
        # Copyright (c) 2009, Nima Moshtagh
        # http://www.mathworks.com/matlabcentral/fileexchange/
        # 9542-minimum-volume-enclosing-ellipsoid
        # Subsequent singular value decomposition of matrix A and and
        # taking the inverse of the square root of the diagonal of the
        # sigma matrix will produce respective semi-axis lengths.
        # Subsequent singular value decomposition of matrix A and
        # taking the inverse of the square root of the diagonal of the
        # sigma matrix will produce respective semi-axis lengths.
        p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                        conv_hull.points[conv_hull.simplices[:, 1], 1],
                        conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
        A, _ = min_vol_ellipse(np.transpose(p), 0.01)
        # New semi-axis lengths
        _, Q, _ = np.linalg.svd(A)
        a = 1/np.sqrt(Q[2])
        b = 1/np.sqrt(Q[1])
        c = 1/np.sqrt(Q[0])
        v_mvee = (4*np.pi*a*b*c)/3
        morph['Fmorph_v_dens_mvee'] = volume / v_mvee

        # Area density - minimum volume enclosing ellipsoid
        # Using a new set of (a,b,c), see Volume density - minimum
        # volume enclosing ellipsoid
        a_mvee = get_area_dens_approx(a, b, c, 20)
        morph['Fmorph_a_dens_mvee'] = area / a_mvee

        # Volume density - convex hull
        v_convex = conv_hull.volume
        morph['Fmorph_v_dens_conv_hull'] = volume / v_convex

        # Area density - convex hull
        a_convex = conv_hull.area
        morph['Fmorph_a_dens_conv_hull'] = area / a_convex

    # Integrated intensity
    if intensity_type == "definite":
        volume = get_mesh_volume(faces, vertices)
        morph['Fmorph_integ_int'] = np.mean(xgl_int) * volume

    # Moran's I index
    if compute_moran_i:
        vol_mor = vol.copy()
        vol_mor[mask_int == 0] = np.NaN
        morph['Fmorph_moran_i'] = get_moran_i(vol_mor, res)

    # Geary's C measure
    if compute_geary_c:
        morph['Fmorph_geary_c'] = get_geary_c(vol_mor, res)

    return morph

def vol(vol: np.ndarray, 
        mask_int: np.ndarray, 
        mask_morph: np.ndarray, 
        res: np.ndarray) -> float:
    """Computes morphological volume feature.
    This feature refers to "Fmorph_vol" (ID = RNUO) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the morphological volume feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)

    return volume  # Morphological volume feature

def approx_vol(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes morphological approximate volume feature.
    This feature refers to "Fmorph_approx_vol" (ID = YEKZ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the morphological approximate volume feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    volume_appro = np.sum(mask_morph[:]) * np.prod(res)

    return volume_appro  # Morphological approximate volume feature

def area(vol: np.ndarray, 
         mask_int: np.ndarray, 
         mask_morph: np.ndarray, 
         res: np.ndarray) -> float:
    """Computes Surface area feature.
    This feature refers to "Fmorph_area" (ID = COJJK) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the surface area feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)

    return area  # Surface area

def av(vol: np.ndarray, 
       mask_int: np.ndarray, 
       mask_morph: np.ndarray, 
       res: np.ndarray) -> float:
    """Computes Surface to volume ratio feature.
    This feature refers to "Fmorph_av" (ID = 2PR5) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Surface to volume ratio feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    ratio = area / volume

    return ratio  # Surface to volume ratio

def comp_1(vol: np.ndarray, 
           mask_int: np.ndarray, 
           mask_morph: np.ndarray, 
           res: np.ndarray) -> float:
    """Computes Compactness 1 feature.
    This feature refers to "Fmorph_comp_1" (ID = SKGS) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Compactness 1 feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    comp_1 = volume / ((np.pi**(1/2))*(area**(3/2)))

    return comp_1  # Compactness 1

def comp_2(vol: np.ndarray, 
           mask_int: np.ndarray, 
           mask_morph: np.ndarray, 
           res: np.ndarray) -> float:
    """Computes Compactness 2 feature.
    This feature refers to "Fmorph_comp_2" (ID = BQWJ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Compactness 2 feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    comp_2 = 36*np.pi*(volume**2) / (area**3)

    return comp_2  # Compactness 2

def sph_dispr(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Spherical disproportion feature.
    This feature refers to "Fmorph_sph_dispr" (ID = KRCK) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Spherical disproportion feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    sph_dispr = area / (36*np.pi*volume**2)**(1/3)

    return sph_dispr  # Spherical disproportion

def sphericity(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Sphericity feature.
    This feature refers to "Fmorph_sphericity" (ID = QCFX) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Sphericity feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    sphericity = ((36*np.pi*volume**2)**(1/3)) / area

    return sphericity  # Sphericity

def asphericity(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Asphericity feature.
    This feature refers to "Fmorph_asphericity" (ID =  25C) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Asphericity feature.
    """
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)
    asphericity = ((area**3) / (36*np.pi*volume**2))**(1/3) - 1

    return asphericity  # Asphericity

def com(vol: np.ndarray, 
        mask_int: np.ndarray, 
        mask_morph: np.ndarray, 
        res: np.ndarray) -> float:
    """Computes Centre of mass shift feature.
    This feature refers to "Fmorph_com" (ID =  KLM) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Centre of mass shift feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, xgl_morph, xyz_int, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    com = get_com(xgl_int, xgl_morph, xyz_int, xyz_morph)

    return com  # Centre of mass shift

def diam(vol: np.ndarray, 
         mask_int: np.ndarray, 
         mask_morph: np.ndarray, 
         res: np.ndarray) -> float:
    """Computes Maximum 3D diameter feature.
    This feature refers to "Fmorph_diam" (ID = L0JK) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Maximum 3D diameter feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, _, _, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    diam = np.max(sc.distance.pdist(conv_hull.points[conv_hull.vertices]))

    return diam  # Maximum 3D diameter

def pca_major(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Major axis length feature.
    This feature refers to "Fmorph_pca_major" (ID = TDIC) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                    XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Major axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, _, _] = get_axis_lengths(xyz_morph)
    pca_major = 4 * np.sqrt(major)

    return pca_major  # Major axis length

def pca_minor(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Minor axis length feature.
    This feature refers to "Fmorph_pca_minor" (ID = P9VJ) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Minor axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [_, minor, _] = get_axis_lengths(xyz_morph)
    pca_minor = 4 * np.sqrt(minor)

    return pca_minor  # Minor axis length

def pca_least(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Least axis length feature.
    This feature refers to "Fmorph_pca_least" (ID = 7J51) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Least axis length feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [_, _, least] = get_axis_lengths(xyz_morph)
    pca_least = 4 * np.sqrt(least)

    return pca_least  # Least axis length

def pca_elongation(vol: np.ndarray, 
                   mask_int: np.ndarray, 
                   mask_morph: np.ndarray, 
                   res: np.ndarray) -> float:
    """Computes Elongation feature.
    This feature refers to "Fmorph_pca_elongation" (ID = Q3CK) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Elongation feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, minor, _] = get_axis_lengths(xyz_morph)
    pca_elongation = np.sqrt(minor / major)

    return pca_elongation  # Elongation

def pca_flatness(vol: np.ndarray, 
                 mask_int: np.ndarray, 
                 mask_morph: np.ndarray, 
                 res: np.ndarray) -> float:
    """Computes Flatness feature.
    This feature refers to "Fmorph_pca_flatness" (ID = N17B) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Flatness feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, _, _, _ = get_variables(vol, mask_int, mask_morph, res)
    [major, _, least] = get_axis_lengths(xyz_morph)
    pca_flatness = np.sqrt(least / major)

    return pca_flatness  # Flatness

def v_dens_aabb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - axis-aligned bounding box feature.
    This feature refers to "Fmorph_v_dens_aabb" (ID = PBX1) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - axis-aligned bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    v_aabb = xc_aabb * yc_aabb * zc_aabb
    v_dens_aabb = volume / v_aabb

    return v_dens_aabb  # Volume density - axis-aligned bounding box

def a_dens_aabb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - axis-aligned bounding box feature.
    This feature refers to "Fmorph_a_dens_aabb" (ID = R59B) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - axis-aligned bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)
    xc_aabb = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    yc_aabb = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    zc_aabb = np.max(vertices[:, 2]) - np.min(vertices[:, 2])
    a_aabb = 2*xc_aabb*yc_aabb + 2*xc_aabb*zc_aabb + 2*yc_aabb*zc_aabb
    a_dens_aabb = area / a_aabb

    return a_dens_aabb  # Area density - axis-aligned bounding box

def v_dens_ombb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - oriented minimum bounding box feature.
    Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
    Determination of the minimum bounding box of an
    arbitrary solid: an iterative approach.
    Comp Struc 79 (2001) 1433-1449.
    This feature refers to "Fmorph_v_dens_ombb" (ID = ZH1A) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - oriented minimum bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    bound_box_dims = min_oriented_bound_box(vertices)
    vol_bb = np.prod(bound_box_dims)
    v_dens_ombb = volume / vol_bb

    return v_dens_ombb  # Volume density - oriented minimum bounding box

def a_dens_ombb(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - oriented minimum bounding box feature.
    Implementation of Chan and Tan's algorithm (C.K. Chan, S.T. Tan.
    Determination of the minimum bounding box of an
    arbitrary solid: an iterative approach.
    This feature refers to "Fmorph_a_dens_ombb" (ID = IQYR) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - oriented minimum bounding box feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)

    bound_box_dims = min_oriented_bound_box(vertices)
    a_ombb = 2 * (bound_box_dims[0] * bound_box_dims[1] 
                + bound_box_dims[0] * bound_box_dims[2]
                + bound_box_dims[1] * bound_box_dims[2])
    a_dens_ombb = area / a_ombb

    return a_dens_ombb  # Area density - oriented minimum bounding box

def v_dens_aee(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Volume density - approximate enclosing ellipsoid feature.
    This feature refers to "Fmorph_v_dens_aee" (ID = 6BDE) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args: 
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - approximate enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    [major, minor, least] = get_axis_lengths(xyz_morph)
    a = 2*np.sqrt(major)
    b = 2*np.sqrt(minor)
    c = 2*np.sqrt(least)
    v_aee = (4*np.pi*a*b*c) / 3
    v_dens_aee = volume / v_aee

    return v_dens_aee  # Volume density - approximate enclosing ellipsoid

def a_dens_aee(vol: np.ndarray, 
               mask_int: np.ndarray, 
               mask_morph: np.ndarray, 
               res: np.ndarray) -> float:
    """Computes Area density - approximate enclosing ellipsoid feature.
    This feature refers to "Fmorph_a_dens_aee" (ID = RDD2) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - approximate enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, xyz_morph, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)
    [major, minor, least] = get_axis_lengths(xyz_morph)
    a = 2*np.sqrt(major)
    b = 2*np.sqrt(minor)
    c = 2*np.sqrt(least)
    a_aee = get_area_dens_approx(a, b, c, 20)
    a_dens_aee = area / a_aee

    return a_dens_aee  # Area density - approximate enclosing ellipsoid

def v_dens_mvee(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Volume density - minimum volume enclosing ellipsoid feature.
    Subsequent singular value decomposition of matrix A and and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    Subsequent singular value decomposition of matrix A and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    This feature refers to "Fmorph_v_dens_mvee" (ID = SWZ1) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - minimum volume enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                    conv_hull.points[conv_hull.simplices[:, 1], 1],
                    conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
    A, _ = min_vol_ellipse(np.transpose(p), 0.01)
    # New semi-axis lengths
    _, Q, _ = np.linalg.svd(A)
    a = 1/np.sqrt(Q[2])
    b = 1/np.sqrt(Q[1])
    c = 1/np.sqrt(Q[0])
    v_mvee = (4*np.pi*a*b*c) / 3
    v_dens_mvee = volume / v_mvee

    return v_dens_mvee  # Volume density - minimum volume enclosing ellipsoid

def a_dens_mvee(vol: np.ndarray, 
                mask_int: np.ndarray, 
                mask_morph: np.ndarray, 
                res: np.ndarray) -> float:
    """Computes Area density - minimum volume enclosing ellipsoid feature.
    Subsequent singular value decomposition of matrix A and and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    Subsequent singular value decomposition of matrix A and
    taking the inverse of the square root of the diagonal of the
    sigma matrix will produce respective semi-axis lengths.
    This feature refers to "Fmorph_a_dens_mvee" (ID = BRI8) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - minimum volume enclosing ellipsoid feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)
    p = np.stack((conv_hull.points[conv_hull.simplices[:, 0], 0],
                    conv_hull.points[conv_hull.simplices[:, 1], 1],
                    conv_hull.points[conv_hull.simplices[:, 2], 2]), axis=1)
    A, _ = min_vol_ellipse(np.transpose(p), 0.01)
    # New semi-axis lengths
    _, Q, _ = np.linalg.svd(A)
    a = 1/np.sqrt(Q[2])
    b = 1/np.sqrt(Q[1])
    c = 1/np.sqrt(Q[0])
    a_mvee = get_area_dens_approx(a, b, c, 20)
    a_dens_mvee = area / a_mvee

    return a_dens_mvee  # Area density - minimum volume enclosing ellipsoid

def v_dens_conv_hull(vol: np.ndarray, 
                     mask_int: np.ndarray, 
                     mask_morph: np.ndarray, 
                     res: np.ndarray) -> float:
    """Computes Volume density - convex hull feature.
    This feature refers to "Fmorph_v_dens_conv_hull" (ID = R3ER) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Volume density - convex hull feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    v_convex = conv_hull.volume
    v_dens_conv_hull = volume / v_convex

    return v_dens_conv_hull  # Volume density - convex hull

def a_dens_conv_hull(vol: np.ndarray, 
                     mask_int: np.ndarray, 
                     mask_morph: np.ndarray, 
                     res: np.ndarray) -> float:
    """Computes Area density - convex hull feature.
    This feature refers to "Fmorph_a_dens_conv_hull" (ID = 7T7F) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Area density - convex hull feature.
    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    _, _, _, _, faces, vertices, conv_hull = get_variables(vol, mask_int, mask_morph, res)
    area = get_mesh_area(faces, vertices)
    v_convex = conv_hull.area
    a_dens_conv_hull = area / v_convex

    return a_dens_conv_hull  # Area density - convex hull

def integ_int(vol: np.ndarray, 
              mask_int: np.ndarray, 
              mask_morph: np.ndarray, 
              res: np.ndarray) -> float:
    """Computes Integrated intensity feature.
    This feature refers to "Fmorph_integ_int" (ID = 99N0) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Value of the Integrated intensity feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)
    xgl_int, _, _, _, faces, vertices, _ = get_variables(vol, mask_int, mask_morph, res)
    volume = get_mesh_volume(faces, vertices)
    integ_int = np.mean(xgl_int) * volume

    return integ_int  # Integrated intensity

def moran_i(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray, 
            res: np.ndarray, 
            compute_moran_i: bool=False) -> float:
    """Computes Moran's I index feature.
    This feature refers to "Fmorph_moran_i" (ID = N365) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).
        compute_moran_i (bool, optional): True to compute Moran's Index.

    Returns:
        float: Value of the Moran's I index feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)

    if compute_moran_i:
        vol_mor = vol.copy()
        vol_mor[mask_int == 0] = np.NaN
        moran_i = get_moran_i(vol_mor, res)

    return moran_i  # Moran's I index

def geary_c(vol: np.ndarray, 
            mask_int: np.ndarray, 
            mask_morph: np.ndarray, 
            res: np.ndarray, 
            compute_geary_c: bool=False) -> float:
    """Computes Geary's C measure feature.
    This feature refers to "Fmorph_geary_c" (ID = NPT7) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        vol (ndarray): 3D volume, NON-QUANTIZED, continous imaging intensity distribution.
        mask_int (ndarray): Intensity mask.
        mask_morph (ndarray): Morphological mask.
        res (ndarray): [a,b,c] vector specfying the resolution of the volume in mm.
                       XYZ resolution (world), or JIK resolution (intrinsic matlab).
        compute_geary_c (bool, optional): True to compute Geary's C measure.

    Returns:
        float: Value of the Geary's C measure feature.

    """
    vol, mask_int, mask_morph = padding(vol, mask_int, mask_morph)

    if compute_geary_c:
        vol_mor = vol.copy()
        vol_mor[mask_int == 0] = np.NaN
        geary_c = get_geary_c(vol_mor, res)

    return geary_c  # Geary's C measure
