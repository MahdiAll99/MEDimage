#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
from scipy.sparse import spdiags

from ..biomarkers.Wei_Toolbox.rle_0 import  rle_0 as rle
from ..biomarkers.Wei_Toolbox.rle_45 import  rle_45 as rle45
from ..biomarkers.Wei_Toolbox.zigzag import  zigzag as zig

def getGLRLMmatrix(roi_only, levels, distCorrection=None) -> np.ndarray:
    """Compute GLRLM matrix.

    This function computes the Gray-Level Run-Length Matrix (GLRLM) of the
    region of interest (ROI) of an input volume. The input volume is assumed
    to be isotropically resampled. Only one GLRLM is computed per scan,
    simultaneously adding up all possible run-lengths in the 13 directions of
    the 3D space. To account for discretization length differences, runs
    constructed from voxels separated by a distance of sqrt(3) increment the
    GLRLM by a value of sqrt(3), runs constructed from voxels separated by a
    distance of sqrt(2) increment the GLRLM by a value of sqrt(2), and runs
    constructed from voxels separated by a distance of 1 increment the GLRLM
    by a value of 1. This function uses other functions from Wei's GLRLM
    toolbox [2].

    Note:
        This function is compatible with 2D analysis (language not adapted in the text).

    Args:
        roi_only_int (ndarray): Smallest box containing the ROI, with the imaging 
            data readyfor texture analysis computations. Voxels outside the ROI 
            are set to NaNs.
        levels (ndarray or List): Vector containing the quantized gray-levels 
            in the tumor region (or reconstruction levels of quantization).
        distCorrection: (optional). Set this variable to true in order to use
            discretization length difference corrections as used
            here: <https://doi.org/10.1088/0031-9155/60/14/5471>.
            Set this variable to false to replicate IBSI results.

    Returns:
        ndarray: Array of Gray-Level Run-Length Matrix of 'ROIOnly'.

    REFERENCES:
        [1] Galloway, M. M. (1975). Texture analysis using gray level run lengths.
            Computer Graphics and Image Processing, 4(2), 172–179.
        [2] Wei's GLRLM toolbox: Xunkai Wei, Gray Level Run Length Matrix Toolbox
            v1.0, Software,Beijing Aeronautical Technology Research Center, 2007.
            <http://www.mathworks.com/matlabcentral/fileexchange/
            17482-gray-level-run-length-matrix-toolbox>

    """

    # PARSING "distCorrection" ARGUMENT
    if (distCorrection is None) or (type(distCorrection) is not bool):
        distCorrection = True  # By default

    if distCorrection:
        factCorr2 = math.sqrt(2)
        factCorr3 = math.sqrt(3)
    else:
        factCorr2 = 1
        factCorr3 = 1

    # PRELIMINARY

    roi_only = roi_only.copy()
    level_temp = np.max(levels)+1
    # Last row needs to be taken out of the GLRLM
    roi_only[np.isnan(roi_only)] = level_temp
    levels = np.append(levels, level_temp)

    # QUANTIZATION EFFECTS CORRECTION

    unique_vol = levels  # levels
    NL = np.size(levels) - 1

    # INITIALIZATION

    size_v = np.shape(roi_only)
    num_init = np.ceil(np.max(size_v)).astype('int')  # Max run length
    GLRLM = np.zeros((NL+1, num_init))

    # START COMPUTATION
    # Directions [1,0,0], [0 1 0], [1 1 0] and [-1 1 0] : 2D directions
    # (x:right-left, y:top-bottom, z:3rd dimension)

    if np.size(size_v) == 3:
        # We can add-up the GLRLMs taken separately in
        # every image in the x-y plane
        n_comp = size_v[2]
    else:
        n_comp = 1

    for i in range(1, n_comp+1):
        image = roi_only[:, :, i-1].copy()
        unique_im = np.unique(image)
        nl_temp = np.size(unique_im)
        index_row = np.zeros(nl_temp).astype('int')
        temp = image.copy()

        for j in range(1, nl_temp+1):
            index_row[j-1] = np.nonzero(unique_im[j-1] == unique_vol)[0]
            image[temp == unique_im[j-1]] = j

        # [1,0,0]
        glrlm_temp = rle.rle_0(image, nl_temp)
        n_run = np.shape(glrlm_temp)[1]
        # Cumulative addition into the GLRLM
        GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp], 0:n_run] +
                                             glrlm_temp[0:nl_temp, 0:n_run])
        # [0 1 0]
        glrlm_temp = rle.rle_0(np.transpose(image), nl_temp)
        n_run = np.shape(glrlm_temp)[1]
        # Cumulative addition into the GLRLM
        GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp], 0:n_run] +
                                             glrlm_temp[0:nl_temp, 0:n_run])

        # [1 1 0]
        seq = zig.zigzag(image)
        glrlm_temp = rle45.rle_45(seq, nl_temp)
        n_run = np.shape(glrlm_temp)[1]
        # Discretisation length difference correction
        GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp], 0:n_run] +
                                             glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

        # [-1 1 0]
        seq = zig.zigzag(np.fliplr(image))
        glrlm_temp = rle45.rle_45(seq, nl_temp)
        n_run = np.shape(glrlm_temp)[1]
        # Discretisation length difference correction
        GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp], 0:n_run] +
                                             glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

    if np.size(size_v) == 3:   # 3D DIRECTIONS
        # Directions [0,0,1], [1 0 1] and [-1 0 1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        # We can add-up the GLRLMs taken separately in every
        # image in the x-z plane
        n_comp = size_v[0]
        image = np.zeros((size_v[2], size_v[1]))

        for i in range(1, n_comp+1):
            for j in range(1, size_v[2]+1):
                image[j-1, :] = roi_only[i-1, :, j-1].copy()

            unique_im = np.unique(image)
            nl_temp = np.size(unique_im)
            index_row = np.zeros(nl_temp).astype('int')
            temp = image.copy()

            for j in range(1, nl_temp+1):
                index_row[j-1] = np.nonzero(unique_im[j-1] == unique_vol)[0]
                image[temp == unique_im[j-1]] = j

            # [0,0,1]
            glrlm_temp = rle.rle_0(np.transpose(image), nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Cumulative addition into the GLRLM
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run])

            # [1 0 1]
            seq = zig.zigzag(image)
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

            # [-1 0 1]
            seq = zig.zigzag(np.fliplr(image))
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

        # Directions [0,1,1] and [0 -1 1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        # We can add-up the GLRLMs taken separately in every
        # image in the y-z plane
        n_comp = size_v[1]
        image = np.zeros((size_v[0], size_v[2]))

        for i in range(1, n_comp+1):
            for j in range(1, size_v[2]+1):
                image[:, j-1] = roi_only[:, i-1, j-1].copy()

            unique_im = np.unique(image)
            nl_temp = np.size(unique_im)
            index_row = np.zeros(nl_temp).astype('int')
            temp = image.copy()

            for j in range(1, nl_temp+1):
                index_row[j-1] = np.nonzero(unique_im[j-1] == unique_vol)[0]
                image[temp == unique_im[j-1]] = j

            # [0,1,1]
            seq = zig.zigzag(image)
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

            # [0 -1 1]
            seq = zig.zigzag(np.fliplr(image))
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr2)

        # Four corners: [1,1,1], [-1,1,1], [-1,1,-1], [1,1,-1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        image = np.zeros((size_v[2], size_v[1]))
        temp = np.random.random((size_v[2], size_v[1]))
        diag_temp = (np.transpose(spdiags(temp, np.arange(
            -(size_v[2]-1), 1), size_v[1]+size_v[2]-1, size_v[1]).todense()))
        sz_diag = np.shape(diag_temp)
        diag_mat1 = np.zeros((sz_diag[0], sz_diag[1], size_v[0]))
        diag_mat2 = np.zeros((sz_diag[0], sz_diag[1], size_v[0]))

        for i in range(1, size_v[0]+1):
            for j in range(1, size_v[2]+1):
                image[j-1, :] = roi_only[i-1, :, j-1].copy()

            try:
                diag_mat1[:, :, i-1] = (np.transpose(spdiags(image, np.arange(-(
                    size_v[2]-1), 1), size_v[1]+size_v[2]-1, size_v[1]).todense()))
            except:
                # Add a column at the beginning to prevent errors
                temp = np.transpose(spdiags(image, np.arange(-(
                    size_v[2]-1), 1), size_v[1]+size_v[2]-1, size_v[1]).todense())
                number_diff = np.abs(np.shape(temp)[1]-np.shape(diag_mat1)[1])
                if np.mod(number_diff, 2):  # Odd difference number
                    temp = np.pad(temp, [0, (number_diff+1)/2], 'constant',
                                  constant_values=0)
                    diag_mat1[:, :, i-1] = temp[:, :-1]
                else:
                    diag_mat1[:, :, i-1] = (np.pad(temp, [0, number_diff/2],
                                                  'constant', constant_values=0))

            try:
                diag_mat2[:, :, i-1] = (np.transpose(spdiags(np.fliplr(image),
                                                            np.arange(-(size_v[2]-1),1),
                                                            size_v[1]+size_v[2]-1,
                                                            size_v[1]).todense()))
            except:
                # Add a column at the beginning to prevent errors
                temp = (np.transpose(spdiags(
                        np.fliplr(image), np.arange(-(size_v[2]-1), 1),
                        size_v[1]+size_v[2]-1, size_v[1]).todense()))
                number_diff = np.abs(np.shape(temp)[1]-np.shape(diag_mat2)[1])
                if np.mod(number_diff, 2):  # Odd difference number
                    temp = np.pad(temp, [0, (number_diff+1)/2],
                                  'constant', constant_values=0)
                    diag_mat2[:, :, i-1] = temp[:, :-1]
                else:
                    diag_mat2[:, :, i-1] = np.pad(temp, [0, number_diff/2],
                                                 'constant', constant_values=0)

        for j in range(1, sz_diag[1]+1):
            index = np.not_equal(diag_mat1[:, j-1, 1], 0).astype('int')
            index_clean = np.nonzero(np.reshape(
                index, np.size(index), order='F') == 1)[0]
            n_temp = np.sum(index)
            image1 = np.zeros((size_v[0], n_temp))
            image2 = np.zeros((size_v[0], n_temp))

            for k in range(1, size_v[0]+1):
                image1[k-1, 0:n_temp] = np.transpose(diag_mat1[index_clean, j-1, k-1])
                image2[k-1, 0:n_temp] = np.transpose(diag_mat2[index_clean, j-1, k-1])

            # 2 first corners
            unique_im = np.unique(image1)
            nl_temp = np.size(unique_im)
            index_row = np.zeros(nl_temp).astype('int')
            temp = image1.copy()
            for i in range(1, nl_temp+1):
                index_row[i-1] = np.nonzero(unique_im[i-1] == unique_vol)[0]
                image1[temp == unique_im[i-1]] = i

            seq = zig.zigzag(image1)
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr3)
            seq = zig.zigzag(np.fliplr(image1))
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr3)

            # 2 last corners
            unique_im = np.unique(image2)
            nl_temp = np.size(unique_im)
            index_row = np.zeros(nl_temp).astype('int')
            temp = image2.copy()
            for i in range(1, nl_temp+1):
                index_row[i-1] = np.nonzero(unique_im[i-1] == unique_vol)[0]
                image2[temp == unique_im[i-1]] = i

            seq = zig.zigzag(image2)
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr3)
            seq = zig.zigzag(np.fliplr(image2))
            glrlm_temp = rle45.rle_45(seq, nl_temp)
            n_run = np.shape(glrlm_temp)[1]
            # Discretisation length difference correction
            GLRLM[index_row[0:nl_temp], 0:n_run] = (GLRLM[index_row[0:nl_temp],
                                                       0:n_run] + glrlm_temp[0:nl_temp, 0:n_run]*factCorr3)

    # REMOVE UNECESSARY COLUMNS

    GLRLM = np.delete(GLRLM, -1, 0)
    stop = np.nonzero(np.sum(GLRLM, 0))[0][-1]
    GLRLM = np.delete(GLRLM, range(stop+1, np.shape(GLRLM)[1]+1), 1)

    return GLRLM
