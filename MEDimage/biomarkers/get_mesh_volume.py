#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def get_mesh_volume(faces, vertices):
    """Compute MeshVolume.
    -------------------------------------------------------------------------
    - faces: [n_points X 3] matrix of three column vectors, defining the [X,Y,Z]
      positions of the faces of the isosurface or convex hull of the mask
      (output from "isosurface.m" or "convhull.m" functions of MATLAB).
      --> These are more precisely indexes to "vertices"
    - vertices: [n_points X 3] matrix of three column vectors, defining the
      [X,Y,Z] positions of the vertices of the isosurface of the mask (output
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

    # Getting vectors for the three vertices
    # (with respect to origin) of each face
    a = vertices[faces[:, 0], :]
    b = vertices[faces[:, 1], :]
    c = vertices[faces[:, 2], :]

    # Calculating volume
    v_cross = np.cross(b, c)
    v_dot = np.sum(a.conj()*v_cross, axis=1)
    volume = np.abs(np.sum(v_dot))/6

    return volume
