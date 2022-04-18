#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getCOM(Xgl_int, Xgl_morph, XYZ_int, XYZ_morph):
    """Compute COM.
    -------------------------------------------------------------------------
    CALCULATE CENTER OF MASS SHIFT (in mm, since "res" is in mm)
     - Xgl: Vector of intensity values in the volume to analyze.
     - XYZ: [nPoints X 3] matrix of three column vectors, defining the [X,Y,Z]
            positions of the points in the ROI (1's) of the mask volume.
            --> In mm.

    IMPORTANT: Row positions of "Xgl" and "XYZ" must correspond for each point.
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
