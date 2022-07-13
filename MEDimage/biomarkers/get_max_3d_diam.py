#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from deprecated import deprecated


@deprecated(reason="Use scipy.distance.pdist() instaed")
def get_max_3d_diam(faces: np.ndarray,
                    vertices: np.ndarray) -> float:
    """Compute Maximum 3D diameter.

    Args:
        faces (ndarray): matrix of three column vectors, defining the
                         [X,Y,Z] positions of the faces of the isosurface or convex hull
                         of the mask (output from "isosurface.m" or "convhull.m" functions of MATLAB).
                         --> These are more precisely indexes to "vertices"
        vertices (ndarray): matrix of three column vectors, defining the
                            [X,Y,Z] positions of the vertices of the isosurface of the
                            mask (in mm)(output from "isosurface.m" function of MATLAB).

    Returns:
        float: Maximum 3D diameter.
    """

    # Finding the max distance between all pair or points of the convex hull
    maxi = 0
    faces = faces.copy()
    vertices = vertices.copy()
    n_points = np.shape(faces)[0]

    for i in range(1, n_points+1):
        for j in range(i+1, n_points+1):
            dist = (vertices[faces[i-1, 0], 0] - vertices[faces[j-1, 0], 0])**2 + (
                    vertices[faces[i-1, 1], 1] - vertices[faces[j-1, 1], 1])**2 + (
                    vertices[faces[i-1, 2], 2] - vertices[faces[j-1, 2], 2])**2

            if dist > maxi:
                maxi = dist

    size_roi = np.sqrt(maxi)

    return size_roi
