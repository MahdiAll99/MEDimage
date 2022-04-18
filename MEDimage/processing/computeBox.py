#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from Code_Radiomics.ImageProcessing.computeBoundingBox import computeBoundingBox
from Code_Utilities.imref import intrinsicToWorld, imref3d


def computeBox(vol, roi, spatialRef, boxString):
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

    # AZ: comments:
    # TODO I would not recommend parsing different settings into a string.
    # Provide two or more parameters instead, and use None if one or more
    # are not used.
    # TODO there is no else statement, so "newSpatialRef" might be unset

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
        Xlimit, Ylimit, Zlimit = intrinsicToWorld(spatialRef, boxBound[0, 0],
                                                  boxBound[1, 0],
                                                  boxBound[2, 0])
        newSpatialRef = imref3d(sizeBox, res[0], res[1], res[2])

        # The limit is defined as the border of the first pixel
        newSpatialRef.XWorldLimits = newSpatialRef.XWorldLimits - (
            newSpatialRef.XWorldLimits[0] - (Xlimit - res[0] / 2))
        newSpatialRef.YWorldLimits = newSpatialRef.YWorldLimits - (
            newSpatialRef.YWorldLimits[0] - (Ylimit - res[1] / 2))
        newSpatialRef.ZWorldLimits = newSpatialRef.ZWorldLimits - (
            newSpatialRef.ZWorldLimits[0] - (Zlimit - res[2] / 2))

    elif "full" in boxString:
        newSpatialRef = spatialRef

    return vol, roi, newSpatialRef
