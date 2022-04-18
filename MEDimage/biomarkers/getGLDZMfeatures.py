#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import Code_Radiomics.ImageBiomarkers.getGLDZMmatrix


def getGLDZMfeatures(volInt, maskMorph):
    """Compute GLDZMfeatures.
    -------------------------------------------------------------------------
     - vol: 3D volume, isotropically resampled, quantized
       (e.g. Ng = 32, levels = [1, ..., Ng]),
        with NaNs outside the region of interest
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

    gldzm = {'Fdzm_sde': [],
             'Fdzm_lde': [],
             'Fdzm_lgze': [],
             'Fdzm_hgze': [],
             'Fdzm_sdlge': [],
             'Fdzm_sdhge': [],
             'Fdzm_ldlge': [],
             'Fdzm_ldhge': [],
             'Fdzm_glnu': [],
             'Fdzm_glnu_norm': [],
             'Fdzm_zdnu': [],
             'Fdzm_zdnu_norm': [],
             'Fdzm_z_perc': [],
             'Fdzm_gl_var': [],
             'Fdzm_zd_var': [],
             'Fdzm_zd_entr': []}

    # GET THE GLDZM MATRIX

    # Correct definition, without any assumption
    levels = np.arange(1, np.max(volInt[~np.isnan(volInt[:])])+1)

    GLDZM = Code_Radiomics.ImageBiomarkers.getGLDZMmatrix.getGLDZMmatrix(
        volInt, maskMorph, levels)
    Ns = np.sum(GLDZM)
    GLDZM = GLDZM/np.sum(GLDZM)  # Normalization of GLDZM
    sz = np.shape(GLDZM)  # Size of GLDZM
    cVect = range(1, sz[1]+1)  # Row vectors
    rVect = range(1, sz[0]+1)  # Column vectors
    # Column and row indicators for each entry of the GLDZM
    cMat, rMat = np.meshgrid(cVect, rVect)
    pg = np.transpose(np.sum(GLDZM, 1))  # Gray-Level Vector
    pd = np.sum(GLDZM, 0)  # Distance Zone Vector

    # COMPUTING TEXTURES

    # Small distance emphasis
    gldzm['Fdzm_sde'] = (np.matmul(pd, np.transpose(np.power(
        1.0/np.array(cVect), 2))))

    # Large distance emphasis
    gldzm['Fdzm_lde'] = (np.matmul(pd, np.transpose(np.power(
        np.array(cVect), 2))))

    # Low grey level zone emphasis
    gldzm['Fdzm_lgze'] = np.matmul(pg, np.transpose(np.power(
        1.0/np.array(rVect), 2)))

    # High grey level zone emphasis
    gldzm['Fdzm_hgze'] = np.matmul(pg, np.transpose(np.power(
        np.array(rVect), 2)))

    # Small distance low grey level emphasis
    gldzm['Fdzm_sdlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/rMat, 2))*(np.power(1.0/cMat, 2))))

    # Small distance high grey level emphasis
    gldzm['Fdzm_sdhge'] = np.sum(np.sum(GLDZM*(np.power(
        rMat, 2))*(np.power(1.0/cMat, 2))))

    # Large distance low grey levels emphasis
    gldzm['Fdzm_ldlge'] = np.sum(np.sum(GLDZM*(np.power(
        1.0/rMat, 2))*(np.power(cMat, 2))))

    # Large distance high grey level emphasis
    gldzm['Fdzm_ldhge'] = np.sum(np.sum(GLDZM*(np.power(
        rMat, 2))*(np.power(cMat, 2))))

    # Gray level non-uniformity
    gldzm['Fdzm_glnu'] = np.sum(np.power(pg, 2)) * Ns

    # Gray level non-uniformity normalised
    gldzm['Fdzm_glnu_norm'] = np.sum(np.power(pg, 2))

    # Zone distance non-uniformity
    gldzm['Fdzm_zdnu'] = np.sum(np.power(pd, 2)) * Ns

    # Zone distance non-uniformity normalised
    gldzm['Fdzm_zdnu_norm'] = np.sum(np.power(pd, 2))

    # Zone percentage
    # Must change the original definition here.
    gldzm['Fdzm_z_perc'] = Ns/np.sum(~np.isnan(volInt[:]))

    # Grey level variance
    temp = rMat * GLDZM
    u = np.sum(temp)
    temp = (np.power(rMat - u, 2)) * GLDZM
    gldzm['Fdzm_gl_var'] = np.sum(temp)

    # Zone distance variance
    temp = cMat * GLDZM
    u = np.sum(temp)
    temp = (np.power(cMat - u, 2)) * GLDZM
    gldzm['Fdzm_zd_var'] = np.sum(temp)

    # Zone distance entropy
    valPos = GLDZM[np.nonzero(GLDZM)]
    temp = valPos * np.log2(valPos)
    gldzm['Fdzm_zd_entr'] = -np.sum(temp)

    return gldzm
