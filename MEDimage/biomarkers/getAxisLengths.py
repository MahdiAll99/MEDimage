#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getAxisLengths(XYZ):
    """Compute AxisLengths.
    -------------------------------------------------------------------------
     - XYZ: [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume.
            --> In mm.
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

    XYZ = XYZ.copy()

    # Getting the geometric centre of mass
    com_geom = np.sum(XYZ, 0)/np.shape(XYZ)[0]  # [1 X 3] vector

    # Subtracting the centre of mass
    XYZ[:, 0] = XYZ[:, 0] - com_geom[0]
    XYZ[:, 1] = XYZ[:, 1] - com_geom[1]
    XYZ[:, 2] = XYZ[:, 2] - com_geom[2]

    # Getting the covariance matrix
    covMat = np.cov(XYZ, rowvar=False)

    # Getting the eigenvalues
    eigVal, _ = np.linalg.eig(covMat)
    eigVal = np.sort(eigVal)
    major = eigVal[2]
    minor = eigVal[1]
    least = eigVal[0]

    return major, minor, least
