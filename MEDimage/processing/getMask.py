#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Radiomics.ImageProcessing.computeBox import computeBox
from Code_Utilities.ImageVolumeObj import ImageVolumeObj
from Code_Utilities.parseContourString import parseContourString


def getMask(sData, contourString, formatData, boxString):
    """
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

    # PARSING OF ARGUMENTS
    if not (boxString == "full" or "box" in boxString):
        raise ValueError(
            "The fourth argument must either be \"full\" or contain the word \"box\".")
    if formatData not in ["nii", "nrrd", "img"]:
        raise ValueError(
            "The third argument must either be \"nii\" or \"nrrd\" or \"img\".")

    contourNumber, operations = parseContourString(contourString)

    # INTIALIZATIONS
    nContour = len(contourNumber)
    # GETTING THE ONE FROM THE DICOM DATA. DICOM DATA THUS MUST BE PRESENT!
    spatialRef = sData[1]['scan']['volume']['spatialRef']
    vol = sData[1][formatData]['volume']['data'].astype(np.float32)

    # COMPUTING ALL MASKS
    ROImask_list = [sData[1][formatData]['mask'][c]
                    ['data'].astype(np.float32) for c in contourNumber]

    # APPLYING OPERATIONS ON ALL MASKS
    roi = ROImask_list[0]
    for c in np.arange(start=1, stop=nContour):
        if operations[c-1] == "+":
            roi += ROImask_list[c]
        elif operations[c-1] == "-":
            roi -= ROImask_list[c]
        else:
            raise ValueError("Unknown operation on ROI.")

        roi[roi >= 1.0] = 1.0
        roi[roi < 1.0] = 0.0

    # COMPUTING THE BOUNDING BOX
    vol, roi, newSpatialRef = computeBox(vol=vol, roi=roi,
                                         spatialRef=spatialRef,
                                         boxString=boxString)

    # ARRANGE OUTPUT
    volObj = ImageVolumeObj(data=vol, spatialRef=newSpatialRef)
    roiObj = ImageVolumeObj(data=roi, spatialRef=newSpatialRef)

    return volObj, roiObj
