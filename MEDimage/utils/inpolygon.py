#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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

import numpy as np


def inpolygon(xq, yq, xv, yv) -> np.ndarray:
    """Implements similar functionality MATLAB inpolygon.
    Finds points located inside or on edge of polygonal region.

    Note: 
        unlike matlab inpolygon, this function does not determine the
        status of single points (xq, yq). Instead, it determines the 
        status for an entire grid by ray-casting.

    Args:
        xq (ndarray): x-coordinates of query points, in intrinsic reference system.
        yq (ndarray): y-coordinates of query points, in intrinsic reference system.
        xq (ndarray): x-coordinates of polygon vertices, in intrinsic reference system.
        yq (ndarray): y-coordinates of polygon vertices, in intrinsic reference system.

    Returns:
        array: boolean array indicating if the query points are on the edge of the polygon area.

    """
    def ray_line_intersection(ray_orig, ray_dir, vert_1, vert_2):

        epsilon = 0.000001

        # Define edge
        edge_line = vert_1 - vert_2

        # Define ray vertices
        r_vert_1 = ray_orig
        r_vert_2 = ray_orig + ray_dir
        edge_ray = - ray_dir

        # Calculate determinant - if close to 0, lines are parallel and will
        # not intersect
        det = np.cross(edge_ray, edge_line)
        if (det > -epsilon) and (det < epsilon):
            return np.nan

        # Calculate inverse of the determinant
        inv_det = 1.0 / det

        # Calculate determinant
        a11 = np.cross(r_vert_1, r_vert_2)
        a21 = np.cross(vert_1, vert_2)

        # Solve for x
        a12 = edge_ray[0]
        a22 = edge_line[0]
        x = np.linalg.det(np.array([[a11, a12], [a21, a22]])) * inv_det

        # Solve for y
        b12 = edge_ray[1]
        b22 = edge_line[1]
        y = np.linalg.det(np.array([[a11, b12], [a21, b22]])) * inv_det

        t = np.array([x, y])

        # Check whether the solution falls within the line segment
        u1 = np.around(np.dot(edge_line, edge_line), 5)
        u2 = np.around(np.dot(edge_line, vert_1-t), 5)
        if (u2 / u1) < 0.0 or (u2 / u1) > 1.0:
            return np.nan

        # Return scalar length from ray origin
        t_scal = np.linalg.norm(ray_orig - t)

        return t_scal

    # These are hacks to actually make this function work
    spacing = np.array([1.0, 1.0])
    origin = np.array([0.0, 0.0])
    shape = np.array([np.max(xq) + 1, np.max(yq) + 1])
    # shape = np.array([np.max(xq), np.max(yq)]) Original from Alex
    vertices = np.vstack((xv, yv)).transpose()
    lines = np.vstack(
        ([np.arange(0, len(xv))], [np.arange(-1, len(xv) - 1)])).transpose()

    # Set up line vertices
    vertex_a = vertices[lines[:, 0], :]
    vertex_b = vertices[lines[:, 1], :]

    # Remove lines with length 0 and center on the origin
    line_mask = np.sum(np.abs(vertex_a - vertex_b), axis=1) > 0.0
    vertex_a = vertex_a[line_mask] - origin
    vertex_b = vertex_b[line_mask] - origin

    # Find extent of contours in x
    x_min_ind = np.int(
        np.max([np.floor(np.min(vertices[:, 0]) / spacing[0]), 0.0]))
    x_max_ind = np.int(
        np.min([np.ceil(np.max(vertices[:, 0]) / spacing[0]), shape[0] * 1.0]))

    # Set up voxel grid and y-span
    vox_grid = np.zeros(shape, dtype=np.int)
    vox_span = origin[1] + np.arange(0, shape[1]) * spacing[1]

    # Set ray origin and direction (starts at negative y, and travels towards
    # positive y
    ray_origin = np.array([0.0, -1.0])
    ray_dir = np.array([0.0, 1.0])

    for x_ind in np.arange(x_min_ind, x_max_ind):
        # Update ray origin
        ray_origin[0] = origin[0] + x_ind * spacing[0]

        # Scan both forward and backward to resolve points located on
        # the polygon
        vox_col_frwd = np.zeros(np.shape(vox_span), dtype=np.int)
        vox_col_bkwd = np.zeros(np.shape(vox_span), dtype=np.int)

        # Find lines that are intersected by the ray
        ray_hit = np.sum(
            np.sign(np.vstack((vertex_a[:, 0], vertex_b[:, 0])) - ray_origin[0]), axis=0)

        # If the ray crosses a vertex, the sum of the sign is 0 when the ray
        # does not hit an vertex point, and -1 or 1 when it does.
        # In the latter case, we only keep of the vertices for each hit.
        simplex_mask = np.logical_or(ray_hit == 0, ray_hit == 1)

        # Go to next iterator if mask is empty
        if np.sum(simplex_mask) == 0:
            continue

        # Determine the selected vertices
        selected_verts = np.squeeze(np.where(simplex_mask))

        # Find intercept of rays with lines
        t_scal = np.array([ray_line_intersection(ray_orig=ray_origin, ray_dir=ray_dir,
                                                 vert_1=vertex_a[ii, :], vert_2=vertex_b[ii, :]) for ii in selected_verts])

        # Remove invalid results
        t_scal = t_scal[np.isfinite(t_scal)]
        if t_scal.size == 0:
            continue

        # Update vox_col based on t_scal. This basically adds a 1 for all
        # voxels that lie behind the line intersections
        # of the ray.
        for t_curr in t_scal:
            vox_col_frwd[vox_span > t_curr + ray_origin[1]] += 1
        for t_curr in t_scal:
            vox_col_bkwd[vox_span < t_curr + ray_origin[1]] += 1

        # Voxels in the roi cross an uneven number of meshes from the origin
        vox_grid[x_ind,
                 :] += np.logical_and(vox_col_frwd % 2, vox_col_bkwd % 2)

    return vox_grid.astype(dtype=np.bool)
