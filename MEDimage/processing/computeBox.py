#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Union

import numpy as np
import computeBoundingBox
from utils.imref import imref3d, intrinsicToWorld
from numpy import ndarray


def computeBox(vol, roi, spatialRef, boxString) -> Union[ndarray, ndarray, imref3d]:
    """Computes a new box around the ROI (Region of interest) from the original box
    and updates the volume and the spatialRef.

    Args:
        vol (ndarray): ROI mask with values of 0 and 1.
        roi (ndarray): ROI mask with values of 0 and 1.
        spatialRef (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        boxString (str): Specifies the new box to be computed
            --> 'full': Full imaging data as output.
            --> 'box' computes the smallest bounding box.
            --> Ex: 'box10': 10 voxels in all three dimensions are added to
                the smallest bounding box. The number after 'box' defines the
                number of voxels to add.
            --> Ex: '2box': Computes the smallest box and outputs double its
                size. The number before 'box' defines the multiplication in
                size.
    Returns:
        ndarray: 3D array of imaging data defining the smallest box containing the ROI.
        ndarray: 3D array of 1's and 0's defining the ROI in ROIbox.
        imref3d: The associated imref3d object imaging data.

    TODO:
        * I would not recommend parsing different settings into a string.
        Provide two or more parameters instead, and use None if one or more
        are not used.
        * there is no else statement, so "newSpatialRef" might be unset
    """    
    if "box" in boxString:
        comp = boxString == "box"
        boxBound = computeBoundingBox(mask=roi)
        if not comp:
            # Always returns the first appearance
            indBox = boxString.find("box")
            # Addition of a certain number of voxels in all dimensions
            if indBox == 0:
                nV = float(boxString[(indBox+3):])
                nV = np.array([nV, nV, nV]).astype(int)
            else:  # Multiplication of the size of the box
                factor = float(boxString[0:indBox])
                sizeBox = np.diff(boxBound, axis=1) + 1
                newBox = sizeBox * factor
                nV = np.round((newBox - sizeBox)/2.0).astype(int)

            ok = False

            while not ok:
                border = np.zeros([3, 2])
                border[0, 0] = boxBound[0, 0] - nV[0]
                border[0, 1] = boxBound[0, 1] + nV[0]
                border[1, 0] = boxBound[1, 0] - nV[1]
                border[1, 1] = boxBound[1, 1] + nV[1]
                border[2, 0] = boxBound[2, 0] - nV[2]
                border[2, 1] = boxBound[2, 1] + nV[2]
                border = border + 1
                check1 = np.sum(border[:, 0] > 0)
                check2 = border[0, 1] <= vol.shape[0]
                check3 = border[1, 1] <= vol.shape[1]
                check4 = border[2, 1] <= vol.shape[2]

                check = check1 + check2 + check3 + check4

                if check == 6:
                    ok = True
                else:
                    nV = np.floor(nV / 2.0)
                    if np.sum(nV) == 0.0:
                        ok = True
                        nV = [0.0, 0.0, 0.0]
        else:
            # Will compute the smallest bounding box possible
            nV = [0.0, 0.0, 0.0]

        boxBound[0, 0] -= nV[0]
        boxBound[0, 1] += nV[0]
        boxBound[1, 0] -= nV[1]
        boxBound[1, 1] += nV[1]
        boxBound[2, 0] -= nV[2]
        boxBound[2, 1] += nV[2]

        boxBound = boxBound.astype(int)

        vol = vol[boxBound[0, 0]:boxBound[0, 1] + 1,
                  boxBound[1, 0]:boxBound[1, 1] + 1,
                  boxBound[2, 0]:boxBound[2, 1] + 1]
        roi = roi[boxBound[0, 0]:boxBound[0, 1] + 1,
                  boxBound[1, 0]:boxBound[1, 1] + 1,
                  boxBound[2, 0]:boxBound[2, 1] + 1]

        # Resolution in mm, nothing has changed here in terms of resolution;
        # XYZ format here.
        res = np.array([spatialRef.PixelExtentInWorldX,
                        spatialRef.PixelExtentInWorldY,
                        spatialRef.PixelExtentInWorldZ])

        # IJK, as required by imref3d
        sizeBox = (np.diff(boxBound, axis=1) + 1).tolist()
        sizeBox[0] = sizeBox[0][0]
        sizeBox[1] = sizeBox[1][0]
        sizeBox[2] = sizeBox[2][0]
        Xlimit, Ylimit, Zlimit = intrinsicToWorld(spatialRef, 
                                                boxBound[0, 0],
                                                boxBound[1, 0],
                                                boxBound[2, 0])
        newSpatialRef = imref3d(sizeBox, res[0], res[1], res[2])

        # The limit is defined as the border of the first pixel
        newSpatialRef.XWorldLimits = newSpatialRef.XWorldLimits - (
            newSpatialRef.XWorldLimits[0] - (Xlimit - res[0]/2))
        newSpatialRef.YWorldLimits = newSpatialRef.YWorldLimits - (
            newSpatialRef.YWorldLimits[0] - (Ylimit - res[1]/2))
        newSpatialRef.ZWorldLimits = newSpatialRef.ZWorldLimits - (
            newSpatialRef.ZWorldLimits[0] - (Zlimit - res[2]/2))

    elif "full" in boxString:
        newSpatialRef = spatialRef

    return vol, roi, newSpatialRef
