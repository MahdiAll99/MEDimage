#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def get_radiomic_names(roi_names, roi_type):
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

    n_names = np.size(roi_names)[0]
    radiomic_names = [0] * n_names
    for n in range(0, n_names):
        radiomic_names[n] = roi_names[n, 0]+'__'+roi_names[n, 1] + \
            '('+roi_type+').'+roi_names[n, 2]+'.npy'

    return radiomic_names
