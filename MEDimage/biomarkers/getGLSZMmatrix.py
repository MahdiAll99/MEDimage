#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import skimage.measure as skim


def getGLSZMmatrix(ROIOnly, levels):
    """Compute GLSZMmatrix.
    -------------------------------------------------------------------------
    getGLSZMmatrix(ROIOnly,levels)
    -------------------------------------------------------------------------
    DESCRIPTION:
    This function computes the Gray-Level Size Zone Matrix (GLSZM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. The zones of different sizes are computed
    using 26-voxel connectivity.

    --> This function is compatible with 2D analysis
        (language not adapted in the text)
    -------------------------------------------------------------------------
    REFERENCE:
    [1] Thibault, G., Fertil, B., Navarro, C., Pereira, S., Cau, P., Levy,
         N., Mari, J.-L. (2009). Texture Indexes and Gray Level Size Zone
         Matrix. Application to Cell Nuclei Classification. In Pattern
         Recognition and Information Processing (PRIP) (pp. 140–145).
    -------------------------------------------------------------------------
    INPUTS:
     - ROIonly: Smallest box containing the ROI, with the imaging data ready
                for texture analysis computations. Voxels outside the ROI are
                set to NaNs.
     - levels: Vector containing the quantized gray-levels in the tumor region
               (or reconstruction levels of quantization).

     ** 'ROIonly' and 'levels' should be outputs from 'prepareVolume.m' **
    -------------------------------------------------------------------------
    OUTPUTS:
     - GLSZM: Gray-Level Size Zone Matrix of 'ROIOnly'.
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

    # PRELIMINARY
    ROIOnly = ROIOnly.copy()
    nMax = np.sum(~np.isnan(ROIOnly))
    levelTemp = np.max(levels) + 1
    ROIOnly[np.isnan(ROIOnly)] = levelTemp
    levels = np.append(levels, levelTemp)

    # QUANTIZATION EFFECTS CORRECTION
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    uniqueVect = levels
    NL = np.size(levels) - 1

    # INITIALIZATION
    # THIS NEEDS TO BE CHANGED. THE ARRAY INITIALIZED COULD BE TOO BIG!
    GLSZM = np.zeros((NL, nMax))

    # COMPUTATION OF GLSZM
    temp = ROIOnly.copy().astype('int')
    for i in range(1, NL+1):
        temp[ROIOnly != uniqueVect[i-1]] = 0
        temp[ROIOnly == uniqueVect[i-1]] = 1
        connObjects, nZone = skim.label(temp, return_num=True)
        for j in range(1, nZone+1):
            col = np.sum(connObjects == j)
            GLSZM[i-1, col-1] = GLSZM[i-1, col-1] + 1

    # REMOVE UNECESSARY COLUMNS
    stop = np.nonzero(np.sum(GLSZM, 0))[0][-1]
    GLSZM = np.delete(GLSZM, range(stop+1, np.shape(GLSZM)[1]), 1)

    return GLSZM
