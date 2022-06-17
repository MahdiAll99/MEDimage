#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def get_glcm_matrix(roi_only, levels, distCorrection=True) -> np.ndarray:
    """Computes GLCM matrix.

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

    Args:
        roi_only (ndarray): Smallest box containing the ROI, with the imaging data 
            ready for texture analysis computations. Voxels outside the ROI are
            set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels in the tumor region
            (or reconstruction levels of quantization).
        distCorrection (bool, optional): Set this variable to true in order to use
            discretization length difference corrections as used
            here: https://doi.org/10.1088/0031-9155/60/14/5471.
            Set this variable to false to replicate IBSI results.

    Returns:
        ndarray: Gray-Level Co-occurence Matrix of `roi_only`.

    REFERENCE:
        [1] Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural
            features for image classification. IEEE Transactions on Systems,
            Man and Cybernetics, smc 3(6), 610–621.
    
    """
    # PARSING "distCorrection" ARGUMENT
    if type(distCorrection) is not bool:
        # The user did not input either "true" or "false",
        # so the default behavior is used.
        distCorrection = True

    # PRELIMINARY
    roi_only = roi_only.copy()
    level_temp = np.max(levels)+1
    roi_only[np.isnan(roi_only)] = level_temp
    #levels = np.append(levels, level_temp)
    dim = np.shape(roi_only)

    if np.ndim(roi_only) == 2:
        dim[2] = 1

    q2 = np.reshape(roi_only, (1, np.prod(dim)))

    # QUANTIZATION EFFECTS CORRECTION (M. Vallieres)
    # In case (for example) we initially wanted to have 64 levels, but due to
    # quantization, only 60 resulted.
    # qs = round(levels*adjust)/adjust;
    # q2 = round(q2*adjust)/adjust;

    #qs = levels
    qs = levels.tolist() + [level_temp]
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
