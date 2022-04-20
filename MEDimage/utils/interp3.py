#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import map_coordinates


def interp3(V, Xq, Yq, Zq, method) -> np.ndarray:
    """Interpolation for 3-D gridded data in meshgrid format,
    implements similar functionality MATLAB interp3.

    REF: <https://www.mathworks.com/help/matlab/ref/interp3.html>

    Args:
        X, Y, Z (ndarray) : Query points, should be intrinsic coordinates.
        method (str): {nearest, linear, spline, cubic}, Interpolation method.

    Returns:
        array: Array of interpolated values.
    
    Raises:
        ValueError: If `method` is not 'nearest', 'linear', 'spline' or 'cubic'.

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
