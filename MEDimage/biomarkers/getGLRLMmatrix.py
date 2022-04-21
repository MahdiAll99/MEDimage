#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
from scipy.sparse import spdiags

from ..biomarkers.Wei_Toolbox.rle_0 import  rle_0 as rle
from ..biomarkers.Wei_Toolbox.rle_45 import  rle_45 as rle45
from ..biomarkers.Wei_Toolbox.zigzag import  zigzag as zig

def getGLRLMmatrix(ROIonly, levels, distCorrection=None) -> np.ndarray:
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
        ROIOnlyInt (ndarray): Smallest box containing the ROI, with the imaging 
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

    ROIonly = ROIonly.copy()
    levelTemp = np.max(levels)+1
    # Last row needs to be taken out of the GLRLM
    ROIonly[np.isnan(ROIonly)] = levelTemp
    levels = np.append(levels, levelTemp)

    # QUANTIZATION EFFECTS CORRECTION

    uniqueVol = levels  # levels
    NL = np.size(levels) - 1

    # INITIALIZATION

    sizeV = np.shape(ROIonly)
    numInit = np.ceil(np.max(sizeV)).astype('int')  # Max run length
    GLRLM = np.zeros((NL+1, numInit))

    # START COMPUTATION
    # Directions [1,0,0], [0 1 0], [1 1 0] and [-1 1 0] : 2D directions
    # (x:right-left, y:top-bottom, z:3rd dimension)

    if np.size(sizeV) == 3:
        # We can add-up the GLRLMs taken separately in
        # every image in the x-y plane
        nComp = sizeV[2]
    else:
        nComp = 1

    for i in range(1, nComp+1):
        image = ROIonly[:, :, i-1].copy()
        uniqueIm = np.unique(image)
        NLtemp = np.size(uniqueIm)
        indexRow = np.zeros(NLtemp).astype('int')
        temp = image.copy()

        for j in range(1, NLtemp+1):
            indexRow[j-1] = np.nonzero(uniqueIm[j-1] == uniqueVol)[0]
            image[temp == uniqueIm[j-1]] = j

        # [1,0,0]
        GLRLMtemp = rle.rle_0(image, NLtemp)
        nRun = np.shape(GLRLMtemp)[1]
        # Cumulative addition into the GLRLM
        GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp], 0:nRun] +
                                             GLRLMtemp[0:NLtemp, 0:nRun])
        # [0 1 0]
        GLRLMtemp = rle.rle_0(np.transpose(image), NLtemp)
        nRun = np.shape(GLRLMtemp)[1]
        # Cumulative addition into the GLRLM
        GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp], 0:nRun] +
                                             GLRLMtemp[0:NLtemp, 0:nRun])

        # [1 1 0]
        seq = zig.zigzag(image)
        GLRLMtemp = rle45.rle_45(seq, NLtemp)
        nRun = np.shape(GLRLMtemp)[1]
        # Discretisation length difference correction
        GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp], 0:nRun] +
                                             GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

        # [-1 1 0]
        seq = zig.zigzag(np.fliplr(image))
        GLRLMtemp = rle45.rle_45(seq, NLtemp)
        nRun = np.shape(GLRLMtemp)[1]
        # Discretisation length difference correction
        GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp], 0:nRun] +
                                             GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

    if np.size(sizeV) == 3:   # 3D DIRECTIONS
        # Directions [0,0,1], [1 0 1] and [-1 0 1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        # We can add-up the GLRLMs taken separately in every
        # image in the x-z plane
        nComp = sizeV[0]
        image = np.zeros((sizeV[2], sizeV[1]))

        for i in range(1, nComp+1):
            for j in range(1, sizeV[2]+1):
                image[j-1, :] = ROIonly[i-1, :, j-1].copy()

            uniqueIm = np.unique(image)
            NLtemp = np.size(uniqueIm)
            indexRow = np.zeros(NLtemp).astype('int')
            temp = image.copy()

            for j in range(1, NLtemp+1):
                indexRow[j-1] = np.nonzero(uniqueIm[j-1] == uniqueVol)[0]
                image[temp == uniqueIm[j-1]] = j

            # [0,0,1]
            GLRLMtemp = rle.rle_0(np.transpose(image), NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Cumulative addition into the GLRLM
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun])

            # [1 0 1]
            seq = zig.zigzag(image)
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

            # [-1 0 1]
            seq = zig.zigzag(np.fliplr(image))
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

        # Directions [0,1,1] and [0 -1 1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        # We can add-up the GLRLMs taken separately in every
        # image in the y-z plane
        nComp = sizeV[1]
        image = np.zeros((sizeV[0], sizeV[2]))

        for i in range(1, nComp+1):
            for j in range(1, sizeV[2]+1):
                image[:, j-1] = ROIonly[:, i-1, j-1].copy()

            uniqueIm = np.unique(image)
            NLtemp = np.size(uniqueIm)
            indexRow = np.zeros(NLtemp).astype('int')
            temp = image.copy()

            for j in range(1, NLtemp+1):
                indexRow[j-1] = np.nonzero(uniqueIm[j-1] == uniqueVol)[0]
                image[temp == uniqueIm[j-1]] = j

            # [0,1,1]
            seq = zig.zigzag(image)
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

            # [0 -1 1]
            seq = zig.zigzag(np.fliplr(image))
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr2)

        # Four corners: [1,1,1], [-1,1,1], [-1,1,-1], [1,1,-1]
        # (x:right-left, y:top-bottom, z:3rd dimension)
        image = np.zeros((sizeV[2], sizeV[1]))
        temp = np.random.random((sizeV[2], sizeV[1]))
        diagTemp = (np.transpose(spdiags(temp, np.arange(
            -(sizeV[2]-1), 1), sizeV[1]+sizeV[2]-1, sizeV[1]).todense()))
        szDiag = np.shape(diagTemp)
        diagMat1 = np.zeros((szDiag[0], szDiag[1], sizeV[0]))
        diagMat2 = np.zeros((szDiag[0], szDiag[1], sizeV[0]))

        for i in range(1, sizeV[0]+1):
            for j in range(1, sizeV[2]+1):
                image[j-1, :] = ROIonly[i-1, :, j-1].copy()

            try:
                diagMat1[:, :, i-1] = (np.transpose(spdiags(image, np.arange(-(
                    sizeV[2]-1), 1), sizeV[1]+sizeV[2]-1, sizeV[1]).todense()))
            except:
                # Add a column at the beginning to prevent errors
                temp = np.transpose(spdiags(image, np.arange(-(
                    sizeV[2]-1), 1), sizeV[1]+sizeV[2]-1, sizeV[1]).todense())
                numberDiff = np.abs(np.shape(temp)[1]-np.shape(diagMat1)[1])
                if np.mod(numberDiff, 2):  # Odd difference number
                    temp = np.pad(temp, [0, (numberDiff+1)/2], 'constant',
                                  constant_values=0)
                    diagMat1[:, :, i-1] = temp[:, :-1]
                else:
                    diagMat1[:, :, i-1] = (np.pad(temp, [0, numberDiff/2],
                                                  'constant', constant_values=0))

            try:
                diagMat2[:, :, i-1] = (np.transpose(spdiags(np.fliplr(image),
                                                            np.arange(-(sizeV[2]-1),1),
                                                            sizeV[1]+sizeV[2]-1,
                                                            sizeV[1]).todense()))
            except:
                # Add a column at the beginning to prevent errors
                temp = (np.transpose(spdiags(
                        np.fliplr(image), np.arange(-(sizeV[2]-1), 1),
                        sizeV[1]+sizeV[2]-1, sizeV[1]).todense()))
                numberDiff = np.abs(np.shape(temp)[1]-np.shape(diagMat2)[1])
                if np.mod(numberDiff, 2):  # Odd difference number
                    temp = np.pad(temp, [0, (numberDiff+1)/2],
                                  'constant', constant_values=0)
                    diagMat2[:, :, i-1] = temp[:, :-1]
                else:
                    diagMat2[:, :, i-1] = np.pad(temp, [0, numberDiff/2],
                                                 'constant', constant_values=0)

        for j in range(1, szDiag[1]+1):
            index = np.not_equal(diagMat1[:, j-1, 1], 0).astype('int')
            index_clean = np.nonzero(np.reshape(
                index, np.size(index), order='F') == 1)[0]
            nTemp = np.sum(index)
            image1 = np.zeros((sizeV[0], nTemp))
            image2 = np.zeros((sizeV[0], nTemp))

            for k in range(1, sizeV[0]+1):
                image1[k-1, 0:nTemp] = np.transpose(diagMat1[index_clean, j-1, k-1])
                image2[k-1, 0:nTemp] = np.transpose(diagMat2[index_clean, j-1, k-1])

            # 2 first corners
            uniqueIm = np.unique(image1)
            NLtemp = np.size(uniqueIm)
            indexRow = np.zeros(NLtemp).astype('int')
            temp = image1.copy()
            for i in range(1, NLtemp+1):
                indexRow[i-1] = np.nonzero(uniqueIm[i-1] == uniqueVol)[0]
                image1[temp == uniqueIm[i-1]] = i

            seq = zig.zigzag(image1)
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr3)
            seq = zig.zigzag(np.fliplr(image1))
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr3)

            # 2 last corners
            uniqueIm = np.unique(image2)
            NLtemp = np.size(uniqueIm)
            indexRow = np.zeros(NLtemp).astype('int')
            temp = image2.copy()
            for i in range(1, NLtemp+1):
                indexRow[i-1] = np.nonzero(uniqueIm[i-1] == uniqueVol)[0]
                image2[temp == uniqueIm[i-1]] = i

            seq = zig.zigzag(image2)
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr3)
            seq = zig.zigzag(np.fliplr(image2))
            GLRLMtemp = rle45.rle_45(seq, NLtemp)
            nRun = np.shape(GLRLMtemp)[1]
            # Discretisation length difference correction
            GLRLM[indexRow[0:NLtemp], 0:nRun] = (GLRLM[indexRow[0:NLtemp],
                                                       0:nRun] + GLRLMtemp[0:NLtemp, 0:nRun]*factCorr3)

    # REMOVE UNECESSARY COLUMNS

    GLRLM = np.delete(GLRLM, -1, 0)
    stop = np.nonzero(np.sum(GLRLM, 0))[0][-1]
    GLRLM = np.delete(GLRLM, range(stop+1, np.shape(GLRLM)[1]+1), 1)

    return GLRLM
