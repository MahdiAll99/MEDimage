#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Radiomics.ImageProcessing.computeBox import computeBox
from Code_Radiomics.ImageProcessing.computeROI import computeROI
from Code_Utilities.ImageVolumeObj import ImageVolumeObj
from Code_Utilities.parseContourString import parseContourString


def getROI(MEDImg, contourString, boxString, interp=None):
    """
    -------------------------------------------------------------------------
    DESCRIPTION:
    Computes the ROI box (+ smallest box containing the region of interest)
    and associated mask from a 'sData' file.
    -------------------------------------------------------------------------
    INPUTS:
    - sData: Cell of structures organizing the data.
    - contourString: In the form '2' or '3-5+3. To be detailed shortly.
    - box:  String specifying the size if the box containing the region
      of interest.
            --> 'full': Full imaging data as output.
            --> 'box' computes the smallest bounding box.
            --> Ex: 'box10': 10 voxels in all three dimensions are added to
                the smallest bounding box. The number after 'box' defines the
                number of voxels to add.
            --> Ex: '2box': Computes the smallest box and outputs double its
                size. The number before 'box' defines the multiplication in
                size.
     - interp: (optional). String specifying if we are to use an interpolation
                  process (using 'interp') prior to "inpolygon.m" in the slice
                  axis direction. See computeROI.m for more details.
            --> Ex: - 'interp': As a consequence: Interpolation is performed
                      in the slice axis dimensions. To be further tested,
                      thus please use with caution. (no interp may be safer)
                    - No argument (default): No interpolation. This can
                      definitely be safe when the RTstruct has been saved
                      specifically for the volume of interest.
    -------------------------------------------------------------------------
    OUTPUTS:
    - volObj: 3D array of imaging data defining the smallest box
              containing the region of interest.
              --> In 'vol' format: vol.data is the 3D array, vol.spatialRef
                  is its associated imref3d object.
    - roiObj: 3D array of 1's and 0's defining the ROI in ROIbox.
              --> In 'vol' format: vol.data is the 3D array, vol.spatialRef
                  is its associated imref3d object.
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
            "The third argument must either be \"full\" or contain the word \"box\".")

    if interp is None:
        interp = "noInterp"
    elif interp != "interp":
        raise ValueError(
            "If present (i.e. it is optional), the fourth argument must be set to \"interp\" ")

    contourNumber, operations = parseContourString(contourString)

    # INTIALIZATIONS
    if type(contourNumber) is int:
        nContour = 1
        contourNumber = [contourNumber]
    else:
        nContour = len(contourNumber)

    ROImask_list = []
    if MEDImg.type in ["PTscan", "CTscan", "MRscan", "ADCscan"]:
        scan = "scan"
        volume = "volume"
    elif MEDImg.type == "PTsim":
        scan = "model"
        volume = "activity"
    elif MEDImg.type == "MRsim":
        # TODO: MRsim-dependent variables have not been implemented.
        raise NotImplementedError(
            "MRsim-dependent variables have not been implemented.")
    else:
        raise ValueError("Unknown scan type.")

    # Note: sData is a nested dictionary not an object
    spatialRef = MEDImg.scan.volume.spatialRef
    vol = MEDImg.scan.volume.data.astype(np.float32)

    # COMPUTING ALL MASKS
    for c in np.arange(start=0, stop=nContour):
        contour = contourNumber[c]
        # GETTING THE XYZ POINTS FROM THE sData STRUCTURE
        ROI_XYZ = MEDImg.scan.contour.points_XYZ[contour].copy()

        # APPLYING ROTATION TO XYZ POINTS (if necessary --> MRscan)
        if hasattr(MEDImg.scan.volume, 'scanRot') and MEDImg.scan.volume.scanRot is not None:
            ROI_XYZ[:, [0, 1, 2]] = np.transpose(
                MEDImg.scan.volume.scanRot @ np.transpose(ROI_XYZ[:, [0, 1, 2]]))

        # APPLYING TRANSLATION IF SIMULATION STRUCTURE AS INPUT
        # (software STAMP utility)
        if hasattr(MEDImg.scan.volume, 'transScanToModel'):
            translation = MEDImg.scan.volume.transScanToModel
            ROI_XYZ[:, 0] += translation[0]
            ROI_XYZ[:, 1] += translation[1]
            ROI_XYZ[:, 2] += translation[2]

        # COMPUTING THE ROI MASK
        # Problem here in computeROI.m: If the volume is a full-body CT and the
        # slice interpolation process occurs, a lot of RAM will be used.
        # One solution could be to a priori compute the bounding box before
        # computing the ROI (using XYZ points). But we still want the user to
        # be able to fully use the "box" argument, so we are fourré...TO SOLVE!
        ROImask_list += [computeROI(ROI_XYZ=ROI_XYZ, spatialRef=spatialRef,
                                    orientation=MEDImg.scan.orientation,
                                    scanType=MEDImg.type,
                                    interp=interp).astype(np.float32)]

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
