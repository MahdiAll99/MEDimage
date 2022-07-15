#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


def min_oriented_bound_box(pos_mat: np.ndarray) -> np.ndarray:
    """Determination of the minimum bounding box of an arbitrary solid: an iterative approach.

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
    sel_row = rot_df.loc[rot_df.vol.idxmin, :]
    ombb_dims = np.array(
        [sel_row.aabb_axis_0, sel_row.aabb_axis_1, sel_row.aabb_axis_2])

    return ombb_dims


def rot_matrix(theta: float,
               dim: int=2,
               rot_axis: int=-1) -> np.ndarray:
    """Creates a 2d or 3d rotation matrix

    Args:
        theta (float): angle in radian
        dim (int, optional): dimension size. Defaults to 2.
        rot_axis (int, optional): rotation axis value. Defaults to -1.

    Returns:
        ndarray: rotation matrix
    """

    if dim == 2:
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

    elif dim == 3:
        if rot_axis == 0:
            rot_mat = np.array([[1.0, 0.0, 0.0],
                                [0.0, np.cos(theta), -np.sin(theta)],
                                [0.0, np.sin(theta), np.cos(theta)]])
        elif rot_axis == 1:
            rot_mat = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                                [0.0, 1.0, 0.0],
                                [-np.sin(theta), 0.0, np.cos(theta)]])
        elif rot_axis == 2:
            rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
        else:
            rot_mat = None
    else:
        rot_mat = None

    return rot_mat


def sig_proc_segmentise(x: List) -> List:
    """Produces a list of segments from input x with values (0,1)

    Args:
        x (List): list of values

    Returns:
        List: list of segments from input x with values (0,1)
    """

    # Create a difference vector
    y = np.diff(x)

    # Find start and end indices of sections with value 1
    ind_1_start = np.array(np.where(y == 1)).flatten()
    if np.shape(ind_1_start)[0] > 0:
        ind_1_start += 1
    ind_1_end = np.array(np.where(y == -1)).flatten()

    # Check for boundary effects
    if x[0] == 1:
        ind_1_start = np.insert(ind_1_start, 0, 0)
    if x[-1] == 1:
        ind_1_end = np.append(ind_1_end, np.shape(x)[0]-1)

    # Generate segment df for segments with value 1
    if np.shape(ind_1_start)[0] == 0:
        df_one = pd.DataFrame({"i":   [],
                               "j":   [],
                               "val": []})
    else:
        df_one = pd.DataFrame({"i":   ind_1_start,
                               "j":   ind_1_end,
                               "val": np.ones(np.shape(ind_1_start)[0])})

    # Find start and end indices for section with value 0
    if np.shape(ind_1_start)[0] == 0:
        ind_0_start = np.array([0])
        ind_0_end = np.array([np.shape(x)[0]-1])
    else:
        ind_0_end = ind_1_start - 1
        ind_0_start = ind_1_end + 1

        # Check for boundary effect
        if x[0] == 0:
            ind_0_start = np.insert(ind_0_start, 0, 0)
        if x[-1] == 0:
            ind_0_end = np.append(ind_0_end, np.shape(x)[0]-1)

        # Check for out-of-range boundary effects
        if ind_0_end[0] < 0:
            ind_0_end = np.delete(ind_0_end, 0)
        if ind_0_start[-1] >= np.shape(x)[0]:
            ind_0_start = np.delete(ind_0_start, -1)

    # Generate segment df for segments with value 0
    if np.shape(ind_0_start)[0] == 0:
        df_zero = pd.DataFrame({"i":   [],
                                "j":   [],
                                "val": []})
    else:
        df_zero = pd.DataFrame({"i":    ind_0_start,
                                "j":    ind_0_end,
                                "val":  np.zeros(np.shape(ind_0_start)[0])})

    df_segm = df_one.append(df_zero).sort_values(by="i").reset_index(drop=True)

    return df_segm


def sig_proc_find_peaks(x: float,
                        ddir: str="pos") -> pd.DataFrame:
    """Determines peak positions in array of values

    Args:
        x (float): value
        ddir (str, optional): positive or negative value. Defaults to "pos".

    Returns:
        pd.DataFrame: peak positions in array of values
    """

    # Invert when looking for local minima
    if ddir == "neg":
        x = -x

    # Generate segments where slope is negative

    df_segm = sig_proc_segmentise(x=(np.diff(x) < 0.0)*1)

    # Start of slope coincides with position of peak (due to index shift induced by np.diff)
    ind_peak = df_segm.loc[df_segm.val == 1, "i"].values

    # Check right boundary
    if x[-1] > x[-2]:
        ind_peak = np.append(ind_peak, np.shape(x)[0]-1)

    # Construct dataframe with index and corresponding value
    if np.shape(ind_peak)[0] == 0:
        df_peak = pd.DataFrame({"ind": [],
                                "val": []})
    else:
        if ddir == "pos":
            df_peak = pd.DataFrame({"ind": ind_peak,
                                    "val": x[ind_peak]})
        if ddir == "neg":
            df_peak = pd.DataFrame({"ind":  ind_peak,
                                    "val": -x[ind_peak]})
    return df_peak
