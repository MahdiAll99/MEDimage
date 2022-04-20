#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np


def getCOM(Xgl_int, Xgl_morph, XYZ_int, XYZ_morph) -> Union[float, np.ndarray]:
    """Calculates center of mass shift (in mm, since resolution is in mm).

    Note: 
        Row positions of "Xgl" and "XYZ" must correspond for each point.
    
    Args:
        Xgl_int (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the intensity mask).
        Xgl_morph (ndarray): Vector of intensity values in the volume to analyze 
            (only values in the morphological mask).
        XYZ_int (ndarray): [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI intensity mesh)
        XYZ_morph (ndarray): [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume (In mm).
            (Mesh-based volume calculated from the ROI morphological mesh)

    Returns:
        Union[float, np.ndarray]: The ROI volume centre of mass.

    """

    # Getting the geometric centre of mass
    Nv = np.size(Xgl_morph)

    com_geom = np.sum(XYZ_morph, 0)/Nv  # [1 X 3] vector

    # Getting the density centre of mass
    XYZ_int[:, 0] = Xgl_int*XYZ_int[:, 0]
    XYZ_int[:, 1] = Xgl_int*XYZ_int[:, 1]
    XYZ_int[:, 2] = Xgl_int*XYZ_int[:, 2]
    com_gl = np.sum(XYZ_int, 0)/np.sum(Xgl_int, 0)  # [1 X 3] vector

    # Calculating the shift
    com = np.linalg.norm(com_geom - com_gl)

    return com
