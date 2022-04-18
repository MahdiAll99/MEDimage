#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np
from skimage.exposure import equalize_hist


def equalization(vol_RE):  # def equalization(vol_RE, Ng=64):
    """
    -------------------------------------------------------------------------
    THIS IS A PURE "WHAT IS CONTAINED WITHIN THE ROI" EQUALIZATION. THIS IS
    NOT INFLUENCED BY THE "userSetMinVal" USED FOR FBS DISCRESTISATION.
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

    # AZ: This was made part of the function call
    # Ng = 64
    # This is the default we will use. It means that when using 'FBS',
    # nQ should be chosen wisely such
    # that the total number of grey levels does not exceed 64, for all
    # patients (recommended).
    # This choice was amde by considering that the best equalization
    # performance for "histeq.m" is obtained with low Ng.
    # WARNING: The effective number of grey levels coming out of "histeq.m"
    # may be lower than Ng.

    # CONSERVE THE INDICES OF THE ROI
    Xgl = np.ravel(vol_RE)
    indROI = np.where(~np.isnan(vol_RE))
    Xgl = Xgl[~np.isnan(Xgl)]

    # ADJUST RANGE BETWEEN 0 and 1
    minVal = np.min(Xgl)
    maxVal = np.max(Xgl)
    Xgl01 = (Xgl - minVal)/(maxVal - minVal)

    # EQUALIZATION
    # Xgl_equal = equalize_hist(Xgl01, nbins=Ng)
    # AT THE MOMENT, WE CHOOSE TO USE THE DEFAULT NUMBER OF BINS OF
    # equalize_hist.py (256)
    Xgl_equal = equalize_hist(Xgl01)
    # RE-ADJUST TO CORRECT RANGE
    Xgl_equal = (Xgl_equal - np.min(Xgl_equal)) / \
        (np.max(Xgl_equal) - np.min(Xgl_equal))
    Xgl_equal = Xgl_equal * (maxVal - minVal)
    Xgl_equal = Xgl_equal + minVal

    # RECONSTRUCT THE VOLUME WITH EQUALIZED VALUES
    volEqual_RE = deepcopy(vol_RE)

    volEqual_RE[indROI] = Xgl_equal

    return volEqual_RE
