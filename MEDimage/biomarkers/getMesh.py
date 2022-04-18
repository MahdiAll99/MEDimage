#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import marching_cubes


def getMesh(mask, res):
    """Compute Mesh.
    -------------------------------------------------------------------------
    - mask: Contains only 0's and 1's.
    - res: [a,b,c] vector specfying the resolution of the volume in mm.
      XYZ resolution (world), or JIK resolution (intrinsic matlab).
    - XYZ: [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
    positions of the points in the ROI (1's) of the mask volume.
    --> In mm.
    - faces: [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
      positions of the faces of the isosurface of the mask (output
      from "isosurface.m" function of MATLAB).
    --> These are more precisely indexes to "vertices".
    - vertices: [nPoints X 3] matrix of three column vectors, defining the
      [X,Y,Z] positions of the vertices of the isosurface of the mask (output
      from "isosurface.m" function of MATLAB).
    --> In mm.

    --> IMPORTANT: Make sure the "mask" is padded with a layer of 0's in all
        dimensions to reduce potential isosurface computation errors.
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
    vertices, faces, norms, values = marching_cubes(
        volume=mask, level=0.5, spacing=res)

    # Getting the X,Y,Z positions of the ROI (i.e. 1's) of the mask
    X = np.reshape(X, (np.size(X), 1), order='F')
    Y = np.reshape(Y, (np.size(Y), 1), order='F')
    Z = np.reshape(Z, (np.size(Z), 1), order='F')

    XYZ = np.concatenate((X, Y, Z), axis=1)
    XYZ = XYZ[np.where(np.reshape(mask, np.size(mask), order='F') == 1)[0], :]

    return XYZ, faces, vertices
