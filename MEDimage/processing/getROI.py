#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from typing import Union

import numpy as np
from utils.ImageVolumeObj import ImageVolumeObj
from utils.parseContourString import parseContourString

import getSepROInames
from processing.computeBox import computeBox
from processing.computeROI import computeROI

_logger = logging.getLogger(__name__)

def getROI(MEDimage, nameROI, boxString, interp=False) -> Union[ImageVolumeObj, ImageVolumeObj]:
    """Computes the ROI box (box containing the region of interest)
    and associated mask from MEDimage object.

    Args:
        MEDimage (object): The MEDimage class object. 
        nameROI (str): name of the ROI since the a volume can have multuiple
            ROIs.
        boxString (str): Specifies the size if the box containing the ROI
            - 'full': Full imaging data as output.
            - 'box' computes the smallest bounding box.
            - Ex: 'box10': 10 voxels in all three dimensions are added to
                the smallest bounding box. The number after 'box' defines the
                number of voxels to add.
            - Ex: '2box': Computes the smallest box and outputs double its
                size. The number before 'box' defines the multiplication in
                size.
        interp (bool): True if we need to use an interpolation for box computation.

    Returns:
        ImageVolumeObj: 3D array of imaging data defining box containing the ROI.
            vol.data is the 3D array, vol.spatialRef is its associated imref3d object.
        ImageVolumeObj: 3D array of 1's and 0's defining the ROI.
            roi.data is the 3D array, roi.spatialRef is its associated imref3d object.

    """
    # PARSING OF ARGUMENTS
    try:
        nameStructureSet = []
        delimiters = ["\+", "\-"]
        nContourData = len(MEDimage.scan.ROI.indexes)

        nameROI, vectPlusMinus = getSepROInames(nameROI, delimiters)
        contourNumber = np.zeros(len(nameROI))

        if nameStructureSet is None:
            nameStructureSet = []

        if nameStructureSet:
            nameStructureSet, _ = getSepROInames(nameStructureSet, delimiters)
            if len(nameROI) != len(nameStructureSet):
                raise ValueError(
                    "The numbers of defined ROI names and Structure Set names are not the same")

        for i in range(0, len(nameROI)):
            for j in range(0, nContourData):
                nameTemp = MEDimage.scan.ROI.get_ROIname(key=j)
                if nameTemp == nameROI[i]:
                    if nameStructureSet:
                        # FOR DICOM + RTSTRUCT
                        nameSetTemp = MEDimage.scan.ROI.get_nameSet(key=j)
                        if nameSetTemp == nameStructureSet[i]:
                            contourNumber[i] = j
                            break
                    else:
                        contourNumber[i] = j
                        break

        nROI = np.size(contourNumber)
        # contourString IS FOR EXAMPLE '3' or '1-3+2'
        contourString = str(contourNumber[0].astype(int))

        for i in range(1, nROI):
            if vectPlusMinus[i-1] == 1:
                sign = '+'
            elif vectPlusMinus[i-1] == -1:
                sign = '-'
            contourString = contourString + sign + \
                str(contourNumber[i].astype(int))

        if not (boxString == "full" or "box" in boxString):
            raise ValueError(
                "The third argument must either be \"full\" or contain the word \"box\".")

        if type(interp) != bool:
            raise ValueError(
                "If present (i.e. it is optional), the fourth argument must be bool")

        contourNumber, operations = parseContourString(contourString)

        # INTIALIZATIONS
        if type(contourNumber) is int:
            nContour = 1
            contourNumber = [contourNumber]
        else:
            nContour = len(contourNumber)

        ROImask_list = []
        if MEDimage.type not in ["PTscan", "CTscan", "MRscan", "ADCscan"]:
            raise ValueError("Unknown scan type.")

        spatialRef = MEDimage.scan.volume.spatialRef
        vol = MEDimage.scan.volume.data.astype(np.float32)

        # COMPUTING ALL MASKS
        for c in np.arange(start=0, stop=nContour):
            contour = contourNumber[c]
            # GETTING THE XYZ POINTS FROM MEDimage
            ROI_XYZ = MEDimage.scan.ROI.get_indexes(key=contour).copy()

            # APPLYING ROTATION TO XYZ POINTS (if necessary --> MRscan)
            if hasattr(MEDimage.scan.volume, 'scanRot') and MEDimage.scan.volume.scanRot is not None:
                ROI_XYZ[:, [0, 1, 2]] = np.transpose(
                    MEDimage.scan.volume.scanRot @ np.transpose(ROI_XYZ[:, [0, 1, 2]]))

            # APPLYING TRANSLATION IF SIMULATION STRUCTURE AS INPUT
            # (software STAMP utility)
            if hasattr(MEDimage.scan.volume, 'transScanToModel'):
                translation = MEDimage.scan.volume.transScanToModel
                ROI_XYZ[:, 0] += translation[0]
                ROI_XYZ[:, 1] += translation[1]
                ROI_XYZ[:, 2] += translation[2]

            # COMPUTING THE ROI MASK
            # Problem here in computeROI.m: If the volume is a full-body CT and the
            # slice interpolation process occurs, a lot of RAM will be used.
            # One solution could be to a priori compute the bounding box before
            # computing the ROI (using XYZ points). But we still want the user to
            # be able to fully use the "box" argument, so we are fourré...TO SOLVE!
            ROImask_list += [computeROI(ROI_XYZ=ROI_XYZ, 
                                        spatialRef=spatialRef,
                                        orientation=MEDimage.scan.orientation,
                                        scanType=MEDimage.type,
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
        vol, roi, newSpatialRef = computeBox(vol=vol, 
                                            roi=roi,
                                            spatialRef=spatialRef,
                                            boxString=boxString)

        # ARRANGE OUTPUT
        volObj = ImageVolumeObj(data=vol, spatialRef=newSpatialRef)
        roiObj = ImageVolumeObj(data=roi, spatialRef=newSpatialRef)

    except Exception as e:
        message = f"\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN getROI(): \n {e}"
        _logger.error(message)

        MEDimage.Params['radiomics']['image'].update(
            {('scale'+(str(MEDimage.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return volObj, roiObj
