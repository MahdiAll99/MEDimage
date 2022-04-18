#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def gclm_DiagProb(p_ij):
    """Compute gclm_DiagProb.
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

    Ng = np.size(p_ij, 0)
    valK = np.arange(0, Ng)
    nK = np.size(valK)
    p_iminusj = np.zeros(nK)

    for iterationK in range(0, nK):
        k = valK[iterationK]
        p = 0
        for i in range(0, Ng):
            for j in range(0, Ng):
                if (k - abs(i-j)) == 0:
                    p += p_ij[i, j]

        p_iminusj[iterationK] = p

    return p_iminusj
