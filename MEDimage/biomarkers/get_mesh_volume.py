#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


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
