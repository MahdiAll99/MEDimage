#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# AUTHOR(S): MEDomicsLab consortium
# -------------------------------------------------------------------------
# STATEMENT:
# This file is part of <https://github.com/MEDomics/MEDomicsLab/>,
# a package providing MATLAB programming tools for radiomics analysis.
#  --> Copyright (C) MEDomicsLab consortium.

# This package is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This package is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this package.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------


import numpy as np


def getPatientNames(roiNames):
    """TODO: Document this function

    Parameters
    ----------

    roiNames: np.ndarray[str]
        Description of the parameter

    Returns
    -------

    list[str]
        Description of return value
    """

    nNames = np.size(roiNames[0])
    patientNames = [0] * nNames
    for n in range(0, nNames):
        patientNames[n] = roiNames[0][n]+'__'+roiNames[1][n] + \
            '.'+roiNames[2][n]+'.npy'

    return patientNames
