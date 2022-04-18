#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getGLCMmatrix(ROIonly, levels, distCorrection=True):
    """Compute GLCMmatrix.
    -------------------------------------------------------------------------
    getGLCMmatrix(ROIonly,levels,distCorrection)
    -------------------------------------------------------------------------
    DESCRIPTION:
    This function computes the Gray-Level Co-occurence Matrix (GLCM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. Only one GLCM is computed per scan,
    simultaneously recording (i.e. adding up) the neighboring properties of
    the 26-connected neighbors of all voxels in the ROI. To account for
    discretization length differences, neighbors at a distance of sqrt(3)
    voxels around a center voxel increment the GLCM by a value of sqrt(3),
    neighbors at a distance of sqrt(2) voxels around a center voxel increment
    the GLCM by a value of sqrt(2), and neighbors at a distance of 1 voxels
    around a center voxel increment the GLCM by a value of 1.

    --> This function is compatible with 2D analysis
    (language not adapted in the text).
    -------------------------------------------------------------------------
    REFERENCE:
    [1] Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural
        features for image classification. IEEE Transactions on Systems,
        Man and Cybernetics, smc 3(6), 610–621.
    -------------------------------------------------------------------------
    INPUTS:
    - ROIonly: Smallest box containing the ROI, with the imaging data ready
               for texture analysis computations. Voxels outside the ROI are
               set to NaNs.
    - levels: Vector containing the quantized gray-levels in the tumor region
              (or reconstruction levels of quantization).
    - distCorrection: (optional). Set this variable to true in order to use
                      discretization length difference corrections as used
                      here: https://doi.org/10.1088/0031-9155/60/14/5471.
                      Set this variable to false to replicate IBSI results.

    ** 'ROIonly' and 'levels' should be outputs from 'prepareVolume.m' **
    -------------------------------------------------------------------------
    OUTPUTS:
    - GLCM: Gray-Level Co-occurence Matrix of 'ROIOnly'.
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

    # PARSING "distCorrection" ARGUMENT

    if type(distCorrection) is not bool:
        # The user did not input either "true" or "false",
        # so the default behavior is used.
        distCorrection = True

    # PRELIMINARY

    ROIonly = ROIonly.copy()
    levelTemp = np.max(levels)+1
    ROIonly[np.isnan(ROIonly)] = levelTemp
    #levels = np.append(levels, levelTemp)
    dim = np.shape(ROIonly)

    if np.ndim(ROIonly) == 2:
        dim[2] = 1

    q2 = np.reshape(ROIonly, (1, np.prod(dim)))

    # QUANTIZATION EFFECTS CORRECTION (M. Vallieres)
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    # qs = round(levels*adjust)/adjust;
    # q2 = round(q2*adjust)/adjust;

    #qs = levels
    qs = levels.tolist() + [levelTemp]
    lqs = np.size(qs)

    q3 = q2*0
    for k in range(0, lqs):
        q3[q2 == qs[k]] = k

    q3 = np.reshape(q3, dim).astype(int)
    GLCM = np.zeros((lqs, lqs))

    for i in range(1, dim[0]+1):
        i_min = max(1, i-1)
        i_max = min(i+1, dim[0])
        for j in range(1, dim[1]+1):
            j_min = max(1, j-1)
            j_max = min(j+1, dim[1])
            for k in range(1, dim[2]+1):
                k_min = max(1, k-1)
                k_max = min(k+1, dim[2])
                val_q3 = q3[i-1, j-1, k-1]
                for I2 in range(i_min, i_max+1):
                    for J2 in range(j_min, j_max+1):
                        for K2 in range(k_min, k_max+1):
                            if (I2 == i) & (J2 == j) & (K2 == k):
                                continue
                            else:
                                val_neighbor = q3[I2-1, J2-1, K2-1]
                                if distCorrection:
                                    # Discretization length correction
                                    GLCM[val_q3, val_neighbor] += \
                                        np.sqrt(abs(I2-i)+abs(J2-j)+abs(K2-k))
                                else:
                                    GLCM[val_q3, val_neighbor] += 1

    GLCM = GLCM[0:-1, 0:-1]

    return GLCM
