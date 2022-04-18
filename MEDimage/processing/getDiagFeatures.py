#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import Code_Radiomics.ImageProcessing.computeBoundingBox


def getDiagFeatures(volObj, roiObj_Int, roiObj_Morph, im_type):
    """Compute getDiagFeatures.
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
    diag = {}

    #  FOR THE IMAGE

    if im_type != 'reSeg':
        # Image dimension x
        diag.update({'image_' + im_type + '_dimX':
                     volObj.spatialRef.ImageSize[0]})

        # Image dimension y
        diag.update({'image_' + im_type + '_dimY':
                     volObj.spatialRef.ImageSize[1]})

        # Image dimension z
        diag.update({'image_' + im_type + '_dimz':
                     volObj.spatialRef.ImageSize[2]})

        # Voxel dimension x
        diag.update({'image_' + im_type + '_voxDimX':
                     volObj.spatialRef.PixelExtentInWorldX})

        # Voxel dimension y
        diag.update({'image_' + im_type + '_voxDimY':
                     volObj.spatialRef.PixelExtentInWorldY})

        # Voxel dimension z
        diag.update({'image_' + im_type + '_voxDimZ':
                     volObj.spatialRef.PixelExtentInWorldZ})

        # Mean intensity
        diag.update({'image_' + im_type + '_meanInt': np.mean(volObj.data)})

        # Minimum intensity
        diag.update({'image_' + im_type + '_minInt': np.min(volObj.data)})

        # Maximum intensity
        diag.update({'image_' + im_type + '_maxInt': np.max(volObj.data)})

    # FOR THE ROI
    boxBound_Int = Code_Radiomics.ImageProcessing.computeBoundingBox.computeBoundingBox(
        roiObj_Int.data)
    boxBound_Morph = Code_Radiomics.ImageProcessing.computeBoundingBox.computeBoundingBox(
        roiObj_Morph.data)
    Xgl_Int = volObj.data[roiObj_Int.data == 1]
    Xgl_Morph = volObj.data[roiObj_Morph.data == 1]

    # Map dimension x
    diag.update({'roi_' + im_type + '_Int_dimX':
                 roiObj_Int.spatialRef.ImageSize[0]})

    # Map dimension y
    diag.update({'roi_' + im_type + '_Int_dimY':
                 roiObj_Int.spatialRef.ImageSize[1]})

    # Map dimension z
    diag.update({'roi_' + im_type + '_Int_dimZ':
                 roiObj_Int.spatialRef.ImageSize[2]})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Int_boxBoundDimX':
                 boxBound_Int[0, 1] - boxBound_Int[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Int_boxBoundDimY':
                 boxBound_Int[1, 1] - boxBound_Int[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Int_boxBoundDimZ':
                 boxBound_Int[2, 1] - boxBound_Int[2, 0] + 1})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimX':
                 boxBound_Morph[0, 1] - boxBound_Morph[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimY':
                 boxBound_Morph[1, 1] - boxBound_Morph[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimZ':
                 boxBound_Morph[2, 1] - boxBound_Morph[2, 0] + 1})

    # Voxel number
    diag.update({'roi_' + im_type + '_Int_voxNumb': np.size(Xgl_Int)})

    # Voxel number
    diag.update({'roi_' + im_type + '_Morph_voxNumb': np.size(Xgl_Morph)})

    # Mean intensity
    diag.update({'roi_' + im_type + '_meanInt': np.mean(Xgl_Int)})

    # Minimum intensity
    diag.update({'roi_' + im_type + '_minInt': np.min(Xgl_Int)})

    # Maximum intensity
    diag.update({'roi_' + im_type + '_maxInt': np.max(Xgl_Int)})

    return diag
