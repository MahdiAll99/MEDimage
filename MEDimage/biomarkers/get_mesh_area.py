#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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
