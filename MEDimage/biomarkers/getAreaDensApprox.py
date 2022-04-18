#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getAreaDensApprox(a, b, c, n):
    """Compute AreaDensApprox.
    -------------------------------------------------------------------------
     - a: Major semi-axis length
     - b: Minor semi-axis length
     - c: Least semi-axis length
     - n: Number of iterations
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

    alpha = np.sqrt(1-b**2/a**2)
    beta = np.sqrt(1-c**2/a**2)
    AB = alpha*beta
    point = (alpha**2 + beta**2) / (2*AB)
    Aell = 0

    for v in range(0, n+1):
        coef = [0]*v + [1]
        legen = np.polynomial.legendre.legval(x=point, c=coef)
        Aell = Aell + AB**v / (1-4*v**2) * legen

    Aell = Aell * 4 * np.pi * a * b

    return Aell
