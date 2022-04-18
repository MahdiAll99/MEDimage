#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getMax3Ddiam(faces, vertices):
    """Compute Max3Ddiam.
    -------------------------------------------------------------------------
     - faces: [nPoints X 3] matrix of three column vectors, defining the
              [X,Y,Z] positions of the faces of the isosurface or convex hull
              of the mask (output from "isosurface.m" or "convhull.m"
              functions of MATLAB).
              --> These are more precisely indexes to "vertices"
     - vertices: [nPoints X 3] matrix of three column vectors, defining the
                 [X,Y,Z] positions of the vertices of the isosurface of the
                 mask (output from "isosurface.m" function of MATLAB).
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

    # Finding the max distance between all pair or points of the convex hull
    maxi = 0
    faces = faces.copy()
    vertices = vertices.copy()
    nPoints = np.shape(faces)[0]

    for i in range(1, nPoints+1):
        for j in range(i+1, nPoints+1):
            dist = (vertices[faces[i-1, 0], 0] - vertices[faces[j-1, 0], 0])**2 + (
                vertices[faces[i-1, 1], 1] - vertices[faces[j-1, 1], 1])**2 + (
                vertices[faces[i-1, 2], 2] - vertices[faces[j-1, 2], 2])**2

            if dist > maxi:
                maxi = dist

    sizeROI = np.sqrt(maxi)

    return sizeROI
