#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


def getLocPeak(imgObj, roiObj, res):
    """Compute LocPeak
    -------------------------------------------------------------------------
    - imgObj: Image object
    - roiObj: ROI objecy
    - res: [X,Y,Z] resolution vector in mm, e.g. [2,2,2]
    This works only in 3D for now.
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

    # INITIALIZATION

    # About 6.2 mm, as defined in document
    distThresh = (3/(4*math.pi))**(1/3)*10

    # Insert -inf outside ROI
    temp = imgObj.copy()
    imgObj = imgObj.copy()
    imgObj[roiObj == 0] = -np.inf

    # Find the location(s) of the maximal voxel
    maxVal = np.max(imgObj)
    I, J, K = np.nonzero(imgObj == maxVal)
    nMax = np.size(I)

    # Reconverting to full object without -Inf
    imgObj = temp

    # Get a meshgrid first
    x = res[0]*(np.arange(imgObj.shape[1])+0.5)
    y = res[1]*(np.arange(imgObj.shape[0])+0.5)
    z = res[2]*(np.arange(imgObj.shape[2])+0.5)
    X, Y, Z = np.meshgrid(x, y, z)  # In mm

    # Calculate the local peak
    maxVal = -np.inf

    for n in range(nMax):
        tempX = X - X[I[n], J[n], K[n]]
        tempY = Y - Y[I[n], J[n], K[n]]
        tempZ = Z - Z[I[n], J[n], K[n]]
        tempDistMesh = (np.sqrt(np.power(tempX, 2) + np.power(tempY, 2) +
                                np.power(tempZ, 2)))
        val = imgObj[tempDistMesh <= distThresh]
        val[np.isnan(val)] = []

        if np.size(val) == 0:
            tempLocalPeak = imgObj[I[n], J[n], K[n]]
        else:
            tempLocalPeak = np.mean(val)
        if tempLocalPeak > maxVal:
            maxVal = tempLocalPeak

    localPeak = maxVal

    return localPeak
