#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np
from skimage.measure import marching_cubes


def getMesh(mask, res) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Mesh.

    Note:
      Make sure the `mask` is padded with a layer of 0's in all
      dimensions to reduce potential isosurface computation errors.

    Args:
        mask (ndarray): Contains only 0's and 1's.
        res (ndarray or List): [a,b,c] vector specfying the resolution of the volume in mm.
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
