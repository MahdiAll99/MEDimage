#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def getNGLDMmatrix(ROIonly, levels) -> np.ndarray:
    """Compute NGLDMmatrix.

    Args:
        ROIOnlyInt (ndarray): Smallest box containing the ROI, with the imaging data ready
            for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).

    Returns:
        ndarray: Array of neighbouring grey level dependence matri of 'ROIOnly'.

    """
    ROIonly = ROIonly.copy()

    # PRELIMINARY
    levelTemp = np.max(levels)+1
    ROIonly[np.isnan(ROIonly)] = levelTemp
    levels = np.append(levels, levelTemp)
    dim = np.shape(ROIonly)
    if np.size(dim) == 2:
        np.append(dim, 1)

    q2 = np.reshape(ROIonly, np.prod(dim), order='F').astype("int")

    # QUANTIZATION EFFECTS CORRECTION (M. Vallieres)
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    # qs = round(levels*adjust)/adjust;
    # q2 = round(q2*adjust)/adjust;
    qs = levels.copy()

    # EL NAQA CODE
    q3 = q2*0
    lqs = np.size(qs)
    for k in range(1, lqs+1):
        q3[q2 == qs[k-1]] = k

    q3 = np.reshape(q3, dim, order='F')

    # Min dependence = 0, Max dependence = 26; So 27 columns
    NGLDM = np.zeros((lqs, 27))
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
                count = 0
                for I2 in range(i_min, i_max+1):
                    for J2 in range(j_min, j_max+1):
                        for K2 in range(k_min, k_max+1):
                            if (I2 == i) & (J2 == j) & (K2 == k):
                                continue
                            else:
                                # a = 0
                                if (val_q3 - q3[I2-1, J2-1, K2-1] == 0):
                                    count += 1

                NGLDM[val_q3-1, count] = NGLDM[val_q3-1, count] + 1

    # Last column was for the NaN voxels, to be removed
    NGLDM = np.delete(NGLDM, -1, 0)
    stop = np.nonzero(np.sum(NGLDM, 0))[0][-1]
    NGLDM = np.delete(NGLDM, range(stop+1, np.shape(NGLDM)[1]+1), 1)

    return NGLDM
