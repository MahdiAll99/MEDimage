#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import numpy as np
from Code_Utilities.imref import imref3d
from Code_Utilities.interp3 import interp3
from Code_Radiomics.ImageProcessing.findSpacing import findSpacing
from Code_Radiomics.ImageProcessing.getPolygonMask import getPolygonMask


def computeROI(ROI_XYZ, spatialRef, orientation, scanType, interp):
    """
    -------------------------------------------------------------------------
    HERE, ONLY THE DIMENSION OF SLICES IS ACTAULLY INTERPOLATED --> THIS IS
    THE ONLY RESOLUTION INFO WE CAN GET FROM THE RTstruct XYZ POINTS.
    WE ASSUME THAT THE FUNCTION "poly2mask.m" WILL CORRECTLY CLOSE ANY
    POLYGON IN THE IN-PLANE DIMENSION, EVEN IF WE GO FROM LOWER TO HIGHER
    RESOLUTION (e.g. RTstruct created on PET and applied to CT)
    --> ALLOWS TO INTERPOLATE A RTstruct CREATED ON ANOTHER IMAGING VOLUME
        WITH DIFFERENT RESOLUTIONS, BUT FROM THE SAME FRAM OF REFERENCE
        (e.g. T1w and T2w in MR scans, PET/CT, etc.)
    --> IN THE IDEAL AND RECOMMENDED CASE, A SPECIFIC RTstruct WAS CREATED AND
        SAVED FOR EACH IMAGING VOLUME (SAFE PRACTICE)
    --> The 'interp' should be used only if tested and verified. 'noInterp'
            is currently the default in getROI.m
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

    # USING INTERPOLATION --> THIS PART NEEDS TO BE FURTHER TESTED.
    # TODO: Consider changing to if statement. Changing interp variable here
    # will change the interp variable everywhere
    interp = deepcopy(interp)

    while interp == "interp":
        # Initialization
        if orientation == "Axial":
            dimIJK = 2
            dimXYZ = 2
            direction = "Z"
            # Only the resolution in 'Z' will be changed
            resXYZ = np.array([spatialRef.PixelExtentInWorldX,
                               spatialRef.PixelExtentInWorldY, 0.0])
        elif orientation == "Sagittal":
            dimIJK = 0
            dimXYZ = 1
            direction = "Y"
            # Only the resolution in 'Y' will be changed
            resXYZ = np.array([spatialRef.PixelExtentInWorldX, 0.0,
                               spatialRef.PixelExtentInWorldZ])
        elif orientation == "Coronal":
            dimIJK = 1
            dimXYZ = 0
            direction = "X"
            # Only the resolution in 'X' will be changed
            resXYZ = np.array([0.0, spatialRef.PixelExtentInWorldY,
                               spatialRef.PixelExtentInWorldZ])
        else:
            raise ValueError(
                "Provided orientation is not one of \"Axial\", \"Sagittal\", \"Coronal\".")

        # Creating new imref3d object for sample points (with slice dimension
        # similar to original volume
        # where RTstruct was created)
        # Slice spacing in mm
        sliceSpacing = findSpacing(
            ROI_XYZ[:, dimIJK], scanType).astype(np.float32)

        # Only one slice found in the function "findSpacing" on the above line.
        # We thus must set "sliceSpacing" to the slice spacing of the queried
        # volume, and no interpolation will be performed.
        if sliceSpacing is None:
            sliceSpacing = spatialRef.PixelExtendInWorld(axis=direction)

        newSize = round(spatialRef.ImageExtentInWorld(
            axis=direction) / sliceSpacing)
        resXYZ[dimXYZ] = sliceSpacing
        sz = spatialRef.ImageSize.copy()
        sz[dimIJK] = newSize

        xWorldLimits = spatialRef.XWorldLimits.copy()
        yWorldLimits = spatialRef.YWorldLimits.copy()
        zWorldLimits = spatialRef.ZWorldLimits.copy()

        newSpatialRef = imref3d(imageSize=sz, pixelExtentInWorldX=resXYZ[0],
                                pixelExtentInWorldY=resXYZ[1],
                                pixelExtentInWorldZ=resXYZ[2],
                                xWorldLimits=xWorldLimits,
                                yWorldLimits=yWorldLimits,
                                zWorldLimits=zWorldLimits)

        diff = (newSpatialRef.ImageExtentInWorld(axis=direction) -
                spatialRef.ImageExtentInWorld(axis=direction))

        if np.abs(diff) >= 0.01:
            # Sampled and queried volume are considered "different".
            newLimit = spatialRef.WorldLimits(axis=direction)[0] - diff / 2.0

            # Sampled volume is now centered on queried volume.
            newSpatialRef.WorldLimits(axis=direction, newValue=(newSpatialRef.WorldLimits(axis=direction) -
                                                                (newSpatialRef.WorldLimits(axis=direction)[0] - newLimit)))
        else:
            # Less than a 0.01 mm, sampled and queried volume are considered
            # to be the same. At this point,
            # spatialRef and newSpatialRef may have differed due to data
            # manipulation, so we simply compute
            # the ROI mask with spatialRef (i.e. simply using "poly2mask.m"),
            # without performing interpolation.
            interp = "noInterp"
            break  # Getting out of the "while" statement

        V = getPolygonMask(ROI_XYZ, newSpatialRef, orientation)

        # Getting query points (Xq,Yq,Zq) of output ROImask
        szQ = spatialRef.ImageSize
        Xqi = np.arange(szQ[0])
        Yqi = np.arange(szQ[1])
        Zqi = np.arange(szQ[2])
        Xqi, Yqi, Zqi = np.meshgrid(Xqi, Yqi, Zqi, indexing='ij')

        # Getting queried mask
        Vq = interp3(V=V, Xq=Xqi, Yq=Yqi, Zq=Zqi, method="cubic")
        ROImask = Vq
        ROImask[Vq < 0.5] = 0
        ROImask[Vq >= 0.5] = 1

        # Getting out of the "while" statement
        interp = 'NoMoreInterp'

    # SIMPLY USING "poly2mask.m" or "inpolygon.m". "inpolygon.m" is slower, but
    # apparently more accurate.
    if interp == "noInterp":
        # Using the inpolygon.m function. To be further tested.
        ROImask = getPolygonMask(ROI_XYZ, spatialRef, orientation)

    return ROImask
