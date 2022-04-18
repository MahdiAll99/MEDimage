#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Utilities.imref import worldToIntrinsic
from Code_Utilities.mode import mode
from Code_Utilities.inpolygon import inpolygon


def getPolygonMask(ROI_XYZ, spatialRef, orientation):
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

    # COMPUTING MASK
    sz = spatialRef.ImageSize.copy()
    ROImask = np.zeros(sz)
    # X,Y,Z in intrinsic image coordinates
    X, Y, Z = worldToIntrinsic(R=spatialRef, xWorld=ROI_XYZ[:, 0],
                               yWorld=ROI_XYZ[:, 1],
                               zWorld=ROI_XYZ[:, 2])

    points = np.transpose(np.vstack((X, Y, Z)))

    if orientation == "Axial":
        a = 0
        b = 1
        c = 2
    elif orientation == "Sagittal":
        a = 1
        b = 2
        c = 0
    elif orientation == "Coronal":
        a = 0
        b = 2
        c = 1
    else:
        raise ValueError(
            "Provided orientation is not one of \"Axial\", \"Sagittal\", \"Coronal\".")

    K = np.round(points[:, c])  # Must assign the points to one slice
    closedContours = np.unique(ROI_XYZ[:, 3])
    xq = np.arange(sz[0])
    yq = np.arange(sz[1])
    xq, yq = np.meshgrid(xq, yq)

    for cc in np.arange(len(closedContours)):
        ind = ROI_XYZ[:, 3] == closedContours[cc]
        # Taking the mode, just in case. But normally, numel(unique(K(ind)))
        # should evaluate to 1, as closed contours are meant to be defined on
        # a given slice
        select_slice = mode(K[ind]).astype(int)
        inpoly = inpolygon(xq=xq, yq=yq, xv=points[ind, a], yv=points[ind, b])
        ROImask[:, :, select_slice] = np.logical_or(
            ROImask[:, :, select_slice], inpoly)

    return ROImask
