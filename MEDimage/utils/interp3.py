#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import map_coordinates


def interp3(V, Xq, Yq, Zq, method):
    """
    Implements similar functionality MATLAB interp3.
    AZ: 1. NOTE: placeholder until I get back to my office
    AZ: 2. NOTE: there is no python function that allows interpolation with
    X, Y, Z as sample points. Well actually,
    there are, but then the set of interpolators is quite limited.
    Xq, Yq and Zq should be intrinsic coordinates
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

    # Parse method
    if method == "nearest":
        spline_order = 0
    elif method == "linear":
        spline_order = 1
    elif method in ["spline", "cubic"]:
        spline_order = 3
    else:
        raise ValueError("Interpolator not implemented.")

    size = np.size(Xq)
    coord_X = np.reshape(Xq, size, order='F')
    coord_Y = np.reshape(Yq, size, order='F')
    coord_Z = np.reshape(Zq, size, order='F')
    coordinates = np.array([coord_X, coord_Y, coord_Z]).astype(np.float32)
    Vq = map_coordinates(input=V.astype(
        np.float32), coordinates=coordinates, order=spline_order, mode='nearest')
    Vq = np.reshape(Vq, np.shape(Xq), order='F')

    return Vq

    # EXAMPLE
    # a = np.arange(12.).reshape((2,3,2), order='F')
    #        0   1   2= Y = 0    1    2
    # X=   0[0., 2., 4.]  [ 6.,  8., 10.]
    # X=   1[1., 3., 5.]  [ 7.,  9., 11.]
    #         Z=0            Z=1
    #                              X1   X2     Y1   Y2     Z1   Z2
    #ndimage.map_coordinates(a, [[0.5, 0.5], [0.5, 1.5], [0.5, 0.5]], order=1)
    #RESULT [4.5, 6.5]
