#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def getRadiomicNames(roiNames, roiType):
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

    nNames = np.size(roiNames)[0]
    radiomicNames = [0] * nNames
    for n in range(0, nNames):
        radiomicNames[n] = roiNames[n, 0]+'__'+roiNames[n, 1] + \
            '('+roiType+').'+roiNames[n, 2]+'.npy'

    return radiomicNames
