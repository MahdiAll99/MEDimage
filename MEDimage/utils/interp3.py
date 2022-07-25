#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.ndimage import map_coordinates


def interp3(v, x_q, y_q, z_q, method) -> np.ndarray:
    """`Interpolation for 3-D gridded data <https://www.mathworks.com/help/matlab/ref/interp3.html>`_\
    in meshgrid format, implements similar functionality MATLAB interp3.

    Args:
        X, Y, Z (ndarray) : Query points, should be intrinsic coordinates.
        method (str): {nearest, linear, spline, cubic}, Interpolation ``method``.

    Returns:
        ndarray: Array of interpolated values.
    
    Raises:
        ValueError: If ``method`` is not 'nearest', 'linear', 'spline' or 'cubic'.

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

    size = np.size(x_q)
    coord_X = np.reshape(x_q, size, order='F')
    coord_Y = np.reshape(y_q, size, order='F')
    coord_Z = np.reshape(z_q, size, order='F')
    coordinates = np.array([coord_X, coord_Y, coord_Z]).astype(np.float32)
    v_q = map_coordinates(input=v.astype(
        np.float32), coordinates=coordinates, order=spline_order, mode='nearest')
    v_q = np.reshape(v_q, np.shape(x_q), order='F')

    return v_q
