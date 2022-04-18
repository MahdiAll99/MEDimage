#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def computeBoundingBox(mask):
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
    -------------------------------------------------------------------------    """

    indices = np.where(np.reshape(mask, np.size(mask), order='F') == 1)
    iV, jV, kV = np.unravel_index(indices, np.shape(mask), order='F')
    boxBound = np.zeros((3, 2))
    boxBound[0, 0] = np.min(iV)
    boxBound[0, 1] = np.max(iV)
    boxBound[1, 0] = np.min(jV)
    boxBound[1, 1] = np.max(jV)
    boxBound[2, 0] = np.min(kV)
    boxBound[2, 1] = np.max(kV)

    return boxBound.astype(int)
