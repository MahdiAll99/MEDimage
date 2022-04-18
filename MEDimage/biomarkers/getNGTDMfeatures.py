#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import Code_Radiomics.ImageBiomarkers.getNGTDMmatrix


def getNGTDMfeatures(vol, distCorrection=None):
    """Compute NGTDMfeatures.
    -------------------------------------------------------------------------
     - vol: 3D volume, isotropically resampled, quantized
       (e.g. Ng = 32, levels = [1, ..., Ng]), with NaNs
       outside the region of interest
     - distCorrection: # Set this variable to true in order to use
       discretization length difference corrections as used here:
       https://doi.org/10.1088/0031-9155/60/14/5471.
       Set this variable to false to replicate IBSI results.
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

    ngtdm = {'Fngt_coarseness': [],
             'Fngt_contrast': [],
             'Fngt_busyness': [],
             'Fngt_complexity': [],
             'Fngt_strength': []}

    # GET THE NGTDM MATRIX
    # Correct definition, without any assumption
    levels = np.arange(1, np.max(vol[~np.isnan(vol[:])].astype("int"))+1)

    if distCorrection is None:
        NGTDM, countValid = Code_Radiomics.ImageBiomarkers.getNGTDMmatrix.getNGTDMmatrix(
            vol, levels)
    else:
        NGTDM, countValid = Code_Radiomics.ImageBiomarkers.getNGTDMmatrix.getNGTDMmatrix(
            vol, levels, distCorrection)

    nTot = np.sum(countValid)
    # Now representing the probability of gray-level occurences
    countValid = countValid/nTot
    NL = np.size(NGTDM)
    Ng = np.sum(countValid != 0)
    pValid = np.where(np.reshape(countValid, np.size(
        countValid), order='F') > 0)[0]+1
    nValid = np.size(pValid)

    # COMPUTING TEXTURES

    # Coarseness
    coarseness = 1/np.matmul(np.transpose(countValid), NGTDM)
    coarseness = min(coarseness, 10**6)
    ngtdm['Fngt_coarseness'] = coarseness

    # Contrast
    if Ng == 1:
        ngtdm['Fngt_contrast'] = 0
    else:
        val = 0
        for i in range(1, NL+1):
            for j in range(1, NL+1):
                val = val + countValid[i-1]*countValid[j-1]*((i-j)**2)
        ngtdm['Fngt_contrast'] = val*np.sum(NGTDM)/(Ng*(Ng-1)*nTot)

    # Busyness
    if Ng == 1:
        ngtdm['Fngt_busyness'] = 0
    else:
        denom = 0
        for i in range(1, nValid+1):
            for j in range(1, nValid+1):
                denom = denom + np.abs(pValid[i-1]*countValid[pValid[i-1]-1] -
                                       pValid[j-1]*countValid[pValid[j-1]-1])
        ngtdm['Fngt_busyness'] = np.matmul(np.transpose(
            countValid), NGTDM)/denom

    # Complexity
    val = 0
    for i in range(1, nValid+1):
        for j in range(1, nValid+1):
            val = val + (np.abs(
                pValid[i-1]-pValid[j-1])/(nTot*(
                    countValid[pValid[i-1]-1] +
                    countValid[pValid[j-1]-1])))*(
                countValid[pValid[i-1]-1]*NGTDM[pValid[i-1]-1] +
                countValid[pValid[j-1]-1]*NGTDM[pValid[j-1]-1])

    ngtdm['Fngt_complexity'] = val

    # Strength
    if np.sum(NGTDM) == 0:
        ngtdm['Fngt_strength'] = 0
    else:
        val = 0
        for i in range(1, nValid+1):
            for j in range(1, nValid+1):
                val = val + (countValid[pValid[i-1]-1] +
                             countValid[pValid[j-1]-1])*(
                    pValid[i-1]-pValid[j-1])**2

        ngtdm['Fngt_strength'] = val/np.sum(NGTDM)

    return ngtdm
