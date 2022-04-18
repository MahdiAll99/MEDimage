#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import Code_Radiomics.ImageBiomarkers.findVX
import Code_Radiomics.ImageBiomarkers.findIX


def getIntVolHistFeatures(vol, wd=None, userSetRange=None):
    """Compute IntVolHistFeatures.
    -------------------------------------------------------------------------
    - vol: 3D volume, QUANTIZED, with NaNs outside the region of interest
    1) Naturally discretised volume can be kept as is (e.g. HU values of
    CT scans)
    2) All other volumes with continuous intensity distribution should be
    quantized (e.g., nBins = 100), with levels = [min, ..., max]

    -> Third argument is optional. It needs to be used when a priori
    discretising with "FBS" or "FBSequal".
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

    # INITIALIZATION

    X = vol[~np.isnan(vol[:])]

    if (vol is not None) & (wd is not None) & (userSetRange is not None):
        if userSetRange:
            minVal = userSetRange[0]
            maxVal = userSetRange[1]
        else:
            minVal = np.min(X)
            maxVal = np.max(X)
    else:
        minVal = np.min(X)
        maxVal = np.max(X)

    if maxVal == np.inf:
        maxVal = np.max(X)

    if minVal == -np.inf:
        minVal = np.min(X)

    # Vector of grey-levels.
    # Values are generated within the half-open interval [minVal,maxVal+wd)
    levels = np.arange(minVal, maxVal+wd, wd)
    Ng = levels.size
    Nv = X.size

    # Initialization of final structure (Dictionary) containing all features.
    intVolHist = {'Fivh_V10': [],
                  'Fivh_V90': [],
                  'Fivh_I10': [],
                  'Fivh_I90': [],
                  'Fivh_V10minusV90': [],
                  'Fivh_I10minusI90': [],
                  'Fivh_auc': []}

    # Calculating fractional volume
    fractVol = np.zeros(Ng)
    for i in range(0, Ng):
        fractVol[i] = 1 - np.sum(X < levels[i])/Nv

    # Calculating intensity fraction
    fractInt = (levels - np.min(levels))/(np.max(levels) - np.min(levels))

    # Volume at intensity fraction 10
    V10 = Code_Radiomics.ImageBiomarkers.findVX.findVX(fractInt, fractVol, 10)
    intVolHist['Fivh_V10'] = V10

    # Volume at intensity fraction 90
    V90 = Code_Radiomics.ImageBiomarkers.findVX.findVX(fractInt, fractVol, 90)
    intVolHist['Fivh_V90'] = V90

    # Intensity at volume fraction 10
    #   For initial arbitrary intensities,
    #   we will always be discretising (1000 bins).
    #   So intensities are definite here.
    I10 = Code_Radiomics.ImageBiomarkers.findIX.findIX(levels, fractVol, 10)
    intVolHist['Fivh_I10'] = I10

    # Intensity at volume fraction 90
    #   For initial arbitrary intensities,
    #   we will always be discretising (1000 bins).
    #   So intensities are definite here.
    I90 = Code_Radiomics.ImageBiomarkers.findIX.findIX(levels, fractVol, 90)
    intVolHist['Fivh_I90'] = I90

    # Volume at intensity fraction difference V10-V90
    intVolHist['Fivh_V10minusV90'] = V10 - V90

    # Intensity at volume fraction difference I10-I90
    #   For initial arbitrary intensities,
    #   we will always be discretising (1000 bins).
    #   So intensities are definite here.
    intVolHist['Fivh_I10minusI90'] = I10 - I90

    # Area under IVH curve
    intVolHist['Fivh_auc'] = np.trapz(fractVol)/(Ng - 1)

    return intVolHist
