#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Utilities.mode import mode


def findSpacing(points, scanType):
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
    README --> This function works for points from at least 2 slices. If only
    one slice is present, the function returns a None.
    -------------------------------------------------------------------------
    """
    decimKeep = 4  # We keep at most 4 decimals to find the slice spacing.

    # Rounding to the nearest 0.1 mm, MRI is more problematic due to arbitrary
    # orientations allowed for imaging volumes.
    if scanType == "MRscan":
        slices = np.unique(np.around(points, 1))
    else:
        slices = np.unique(np.around(points, 2))

    nSlices = len(slices)
    if nSlices == 1:
        return None

    diff = np.abs(np.diff(slices))
    diff = np.round(diff, decimKeep)
    sliceSpacing, nOcc = mode(x=diff, return_counts=True)
    if np.max(nOcc) == 1:
        sliceSpacing = np.mean(diff)

    return sliceSpacing
