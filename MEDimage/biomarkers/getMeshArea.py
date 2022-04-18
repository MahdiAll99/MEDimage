#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getMeshArea(faces, vertices):
    """Compute MeshArea.
    -------------------------------------------------------------------------
    - faces: [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the faces of the isosurface or convex hull of the mask
            (output from "isosurface.m" or "convhull.m" functions of MATLAB).
              --> These are more precisely indexes to "vertices"
    - vertices: [nPoints X 3] matrix of three column vectors,
            defining the [X,Y,Z]
            positions of the vertices of the isosurface of the mask (output
            from "isosurface.m" function of MATLAB).
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

    faces = faces.copy()
    vertices = vertices.copy()

    # Getting two vectors of edges for each face
    a = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    b = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]

    # Calculating the surface area of each face and summing it up all at once.
    c = np.cross(a, b)
    area = 1/2 * np.sum(np.sqrt(np.sum(np.power(c, 2), 1)))

    return area
