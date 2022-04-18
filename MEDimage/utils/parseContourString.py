#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from Code_Utilities.strfind import strfind


def parseContourString(contourString):
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

    if isinstance(contourString, (int, float)):
        return contourString, []

    indPlus = strfind(string=contourString, pattern='\+')
    indMinus = strfind(string=contourString, pattern='\-')
    indOperations = np.sort(np.hstack((indPlus, indMinus))).astype(int)

    # Parsing operations and contour numbers
    # AZ: I assume that contourNumber is an integer
    if indOperations.size == 0:
        operations = []

        contourNumber = [int(contourString)]
    else:
        nOp = len(indOperations)
        operations = [contourString[indOperations[i]] for i in np.arange(nOp)]

        contourNumber = np.zeros(nOp + 1, dtype=int)
        contourNumber[0] = int(contourString[0:indOperations[0]])
        for c in np.arange(start=1, stop=nOp):
            contourNumber[c] = int(
                contourString[(indOperations[c-1]+1):indOperations[c]])
        contourNumber[-1] = int(contourString[(indOperations[-1]+1):])
        contourNumber.tolist()

    return contourNumber, operations
