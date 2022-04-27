#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from typing import Tuple

import numpy as np

from ..processing.computeBox import computeBox
from ..utils.ImageVolumeObj import ImageVolumeObj
from ..utils.parseContourString import parseContourString
from .getSepROInames import getSepROInames

_logger = logging.getLogger(__name__)

def getROI_fromIndexes(MEDimg, nameROI, boxString) -> Tuple[ImageVolumeObj, ImageVolumeObj]:
        """Extracts the ROI box (+ smallest box containing the region of interest)
        and associated mask from the indexes saved in 'MEDimage' file.
        
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

        Returns:
            ndarray: volObj, 3D array of imaging data defining the smallest box
                containing the region of interest.
            ndarray: roiObj, 3D array of 1's and 0's defining the ROI in ROIbox.

        """
        # This takes care of the "Volume resection" step
        # as well using the argument "box". No fourth
        # argument means 'interp' by default.

        # PARSING OF ARGUMENTS
        try:
            nameStructureSet = []
            delimiters = ["\+", "\-"]
            nContourData = len(MEDimg.scan.ROI.indexes)

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
                    nameTemp = MEDimg.scan.ROI.get_ROIname(key=j)
                    if nameTemp == nameROI[i]:
                        if nameStructureSet:
                            # FOR DICOM + RTSTRUCT
                            nameSetTemp = MEDimg.scan.ROI.get_nameSet(key=j)
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

            contourNumber, operations = parseContourString(contourString)

            # INTIALIZATIONS
            if type(contourNumber) is int:
                nContour = 1
                contourNumber = [contourNumber]
            else:
                nContour = len(contourNumber)

            # Note: sData is a nested dictionary not an object
            spatialRef = MEDimg.scan.volume.spatialRef
            vol = MEDimg.scan.volume.data.astype(np.float32)

            # APPLYING OPERATIONS ON ALL MASKS
            roi = MEDimg.scan.get_indexes_by_ROIname(nameROI[0])
            for c in np.arange(start=1, stop=nContour):
                if operations[c-1] == "+":
                    roi += MEDimg.scan.get_indexes_by_ROIname(nameROI[c])
                elif operations[c-1] == "-":
                    roi -= MEDimg.scan.get_indexes_by_ROIname(nameROI[c])
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

        except Exception as e:
            message = "\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN getROI_fromIndexes(): " \
                    "\n {}".format(e)
            _logger.error(message)
            print(message)

            MEDimg.Params['radiomics']['image'].update(
                {('scale'+(str(MEDimg.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})
            
            MEDimg.Continue=True

        return volObj, roiObj
