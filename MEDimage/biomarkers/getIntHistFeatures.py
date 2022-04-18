#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import numpy as np
from scipy.stats import scoreatpercentile, variation


def getIntHistFeatures(vol):
    """Compute IntHistFeatures.
    -------------------------------------------------------------------------
    - vol: 3D volume, QUANTIZED (e.g. nBins = 100, levels = [1, ..., max]),
           with NaNs outside the region of interest.
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
    warnings.simplefilter("ignore")

    # INITIALIZATION

    X = vol[~np.isnan(vol[:])]
    Nv = X.size

    intHist = {'Fih_mean': [],
               'Fih_var': [],
               'Fih_skew': [],
               'Fih_kurt': [],
               'Fih_median': [],
               'Fih_min': [],
               'Fih_P10': [],
               'Fih_P90': [],
               'Fih_max': [],
               'Fih_mode': [],
               'Fih_iqr': [],
               'Fih_range': [],
               'Fih_mad': [],
               'Fih_rmad': [],
               'Fih_medad': [],
               'Fih_cov': [],
               'Fih_qcod': [],
               'Fih_entropy': [],
               'Fih_uniformity': [],
               'Fih_max_grad': [],
               'Fih_max_grad_gl': [],
               'Fih_min_grad': [],
               'Fih_min_grad_gl': []}

    # CONSTRUCTION OF HISTOGRAM AND ASSOCIATED NUMBER OF GRAY-LEVELS

    # Always defined from 1 to the maximum value of
    # the volume to remove any ambiguity
    levels = np.arange(1, np.max(X)+100*np.finfo(float).eps)
    Ng = levels.size  # Number of gray-levels
    H = np.zeros(Ng)  # The histogram of X

    for i in range(0, Ng):
        # == i or == levels(i) is equivalent since levels = 1:max(X),
        # and Ng = numel(levels)
        H[i] = np.sum(X == i+1)  # H[i] = sum(X == i+1)

    p = (H/Nv)  # Occurence probability for each grey level bin i
    pt = p.transpose()

    # STARTING COMPUTATION

    # Intensity histogram mean
    u = np.matmul(levels, pt)
    intHist['Fih_mean'] = u

    # Intensity histogram variance
    var = np.matmul(np.power(levels - u, 2), pt)
    intHist['Fih_var'] = var

    # Intensity histogram skewness and kurtosis
    skew = 0
    kurt = 0

    if var != 0:
        skew = np.matmul(np.power(levels - u, 3), pt)/np.power(var, 3/2)
        kurt = np.matmul(np.power(levels - u, 4), pt)/np.power(var, 2) - 3

    intHist['Fih_skew'] = skew
    intHist['Fih_kurt'] = kurt

    # Intensity histogram median
    intHist['Fih_median'] = np.median(X)

    # Intensity histogram minimum grey level
    intHist['Fih_min'] = np.min(X)

    # Intensity histogram 10th percentile
    p10 = scoreatpercentile(X, 10)
    intHist['Fih_P10'] = p10

    # Intensity histogram 90th percentile
    p90 = scoreatpercentile(X, 90)
    intHist['Fih_P90'] = p90

    # Intensity histogram maximum grey level
    intHist['Fih_max'] = np.max(X)

    # Intensity histogram mode
    #    levels = 1:max(X), so the index of the ith bin of H is the same as i
    mH = np.max(H)
    mode = np.where(H == mH)[0] + 1

    if np.size(mode) > 1:
        dist = np.abs(mode - u)
        indMin = np.argmin(dist)
        intHist['Fih_mode'] = mode[indMin]
    else:
        intHist['Fih_mode'] = mode[0]

    # Intensity histogram interquantile range
    #    Since X goes from 1:max(X), all with integer values,
    #    the result is an integer
    intHist['Fih_iqr'] = scoreatpercentile(X, 75)-scoreatpercentile(X, 25)

    # Intensity histogram range
    intHist['Fih_range'] = np.max(X) - np.min(X)

    # Intensity histogram mean absolute deviation
    intHist['Fih_mad'] = np.mean(abs(X - u))

    # Intensity histogram robust mean absolute deviation
    X_10_90 = X[np.where((X >= p10) & (X <= p90), True, False)]
    intHist['Fih_rmad'] = np.mean(np.abs(X_10_90 - np.mean(X_10_90)))

    # Intensity histogram median absolute deviation
    intHist['Fih_medad'] = np.mean(np.absolute(X - np.median(X)))

    # Intensity histogram coefficient of variation
    intHist['Fih_cov'] = variation(X)

    # Intensity histogram quartile coefficient of dispersion
    X_75_25 = scoreatpercentile(X, 75)+scoreatpercentile(X, 25)
    intHist['Fih_qcod'] = intHist['Fih_iqr']/X_75_25

    # Intensity histogram entropy
    p = p[p > 0]
    intHist['Fih_entropy'] = -np.sum(p*np.log2(p))

    # Intensity histogram uniformity
    intHist['Fih_uniformity'] = np.sum(np.power(p, 2))

    # Calculation of histogram gradient
    histGrad = np.zeros(Ng)
    histGrad[0] = H[1] - H[0]
    histGrad[-1] = H[-1] - H[-2]

    for i in range(1, Ng-1):
        histGrad[i] = (H[i+1] - H[i-1])/2

    # Maximum histogram gradient
    intHist['Fih_max_grad'] = np.max(histGrad)

    # Maximum histogram gradient grey level
    indMax = np.where(histGrad == intHist['Fih_max_grad'])[0][0]
    intHist['Fih_max_grad_gl'] = levels[indMax]

    # Minimum histogram gradient
    intHist['Fih_min_grad'] = np.min(histGrad)

    # Minimum histogram gradient grey level
    indMin = np.where(histGrad == intHist['Fih_min_grad'])[0][0]
    intHist['Fih_min_grad_gl'] = levels[indMin]

    return intHist
