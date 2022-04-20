#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import logging

import numpy as np
from utils.ImageVolumeObj import ImageVolumeObj
from utils.imref import imref3d, intrinsicToWorld, worldToIntrinsic
from utils.interp3 import interp3

from processing.computeBox import computeBox

_logger = logging.getLogger(__name__)


def interpVolume(MEDimage, 
                volObjS, 
                voxDim=None, 
                interpMet=None, 
                roundVal=None,
                image_type=None, 
                roiObjS=None, 
                boxString=None) -> ImageVolumeObj:
    """3D voxel interpolation on the input volume.

    Args:
        MEDimage (object): The MEDimage class object.
        volObjS (ImageVolumeObj): Imaging  that will be interpolated.
        voxDim (array): Array of the voxel dimension. The following format is used 
            [Xin,Yin,Zslice], where Xin and Yin are the X (left to right) and 
            Y (bottom to top) IN-PLANE resolutions, and Zslice is the slice spacing,
            NO MATTER THE ORIENTATION OF THE VOLUME (i.e. axial , sagittal, coronal).
        interpMet (str): {nearest, linear, spline, cubic} optional, Interpolation method
        roundVal (float): Rounding value. Must be between 0 and 1 for ROI interpolation
            and to a power of 10 for Image interpolation.
        image_type (str): 'image' for imaging data interpolation and 'roi' for ROI mask
            data interpolation.
        roiObjS (ImageVolumeObj): Mask data, will be used to compute a new specific box 
            and the new imref3d object for the imaging data.
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
        ndarray: 3D array of 1's and 0's defining the ROI mask.

    """
    try:
        # PARSING ARGUMENTS
        if voxDim is None:
            return deepcopy(volObjS)
        if np.sum(voxDim) == 0:
            return deepcopy(volObjS)
        if len(voxDim) == 2:
            twoD = True
        else:
            twoD = False

        if interpMet is None:
            raise ValueError("Interpolation method should be provided.")

        if image_type is None:
            raise ValueError(
                "The type of input image should be specified as \"image\" or \"roi\".")
        if image_type not in ["image", "roi"]:
            raise ValueError(
                "The type of input image should either be \"image\" or \"roi\".")

        if image_type == "image":
            if interpMet not in ["linear", "cubic", "spline"]:
                raise ValueError(
                    "Interpolation method for images should either be \"linear\", \"cubic\" or \"spline\".")
            if roundVal is not None:
                if np.mod(np.log10(roundVal), 1):
                    raise ValueError("\"roundVal\" should be a power of 10.")
        else:
            if interpMet not in ["nearest", "linear", "cubic"]:
                raise ValueError(
                    "Interpolation method for images should either be \"nearest\", \"linear\" or \"cubic\".")
            if roundVal is not None:
                if roundVal < 0.0 or roundVal > 1.0:
                    raise ValueError("\"roundVal\" must be between 0.0 and 1.0.")
            else:
                raise ValueError("\"roundVal\" must be provided for \"roi\".")

        if roiObjS is None or boxString is None:
            useBox = False
        else:
            useBox = True

        # --> QUERIED POINTS: NEW INTERPOLATED VOLUME: "q" or "Q".
        # --> SAMPLED POINTS: ORIGINAL VOLUME: "s" or "S".
        # --> Always using XYZ coordinates (unless specifically noted),
        #     not MATLAB IJK, so beware!

        # INITIALIZATION
        resQ = voxDim
        if twoD:
            # If 2D, the resolution of the slice dimension of he queried volume is
            # set to the same as the sampled volume.
            resQ = np.concatenate((resQ, volObjS.spatialRef.PixelExtentInWorldZ))

        resS = np.array([volObjS.spatialRef.PixelExtentInWorldX,
                        volObjS.spatialRef.PixelExtentInWorldY,
                        volObjS.spatialRef.PixelExtentInWorldZ])

        if np.array_equal(resS, resQ):
            return deepcopy(volObjS)

        spatialRefS = volObjS.spatialRef
        extentS = np.array([spatialRefS.ImageExtentInWorldX,
                            spatialRefS.ImageExtentInWorldY,
                            spatialRefS.ImageExtentInWorldZ])
        lowLimitsS = np.array([spatialRefS.XWorldLimits[0],
                            spatialRefS.YWorldLimits[0],
                            spatialRefS.ZWorldLimits[0]])

        # CREATING QUERIED "imref3d" OBJECT CENTERED ON SAMPLED VOLUME

        # Switching to IJK (matlab) reference frame for "imref3d" computation.
        # Putting a "ceil", according to IBSI standards. This is safer than "round".
        sizeQ = np.ceil(np.around(np.divide(extentS, resQ),
                                decimals=3)).astype(int).tolist()

        if twoD:
            # If 2D, forcing the size of the queried volume in the slice dimension
            # to be the same as the sample volume.
            sizeQ[2] = volObjS.spatialRef.ImageSize[2]

        spatialRefQ = imref3d(imageSize=sizeQ, 
                            pixelExtentInWorldX=resQ[0],
                            pixelExtentInWorldY=resQ[1],
                            pixelExtentInWorldZ=resQ[2])

        extentQ = np.array([spatialRefQ.ImageExtentInWorldX,
                            spatialRefQ.ImageExtentInWorldY,
                            spatialRefQ.ImageExtentInWorldZ])
        lowLimitsQ = np.array([spatialRefQ.XWorldLimits[0],
                            spatialRefQ.YWorldLimits[0],
                            spatialRefQ.ZWorldLimits[0]])
        diff = extentQ - extentS
        newLowLimitsQ = lowLimitsS - diff/2
        spatialRefQ.XWorldLimits = spatialRefQ.XWorldLimits - \
            (lowLimitsQ[0] - newLowLimitsQ[0])
        spatialRefQ.YWorldLimits = spatialRefQ.YWorldLimits - \
            (lowLimitsQ[1] - newLowLimitsQ[1])
        spatialRefQ.ZWorldLimits = spatialRefQ.ZWorldLimits - \
            (lowLimitsQ[2] - newLowLimitsQ[2])

        # REDUCE THE SIZE OF THE VOLUME PRIOR TO INTERPOLATION
        # TODO check that computeBox vol and roi are intended to be the same!
        if useBox:
            _, _, tempSpatialRef = computeBox(
                vol=roiObjS.data, roi=roiObjS.data, spatialRef=volObjS.spatialRef,
                boxString=boxString)

            sizeTemp = tempSpatialRef.ImageSize

            # Getting world boundaries (center of voxels) of the new box
            Xbound, Ybound, Zbound = intrinsicToWorld(R=tempSpatialRef,
                                                    xIntrinsic=np.array(
                                                        [0.0, sizeTemp[0]-1.0]),
                                                    yIntrinsic=np.array(
                                                        [0.0, sizeTemp[1]-1.0]),
                                                    zIntrinsic=np.array([0.0, sizeTemp[2]-1.0]))

            # Getting the image positions of the boundaries of the new box, IN THE
            # FULL QUERIED FRAME OF REFERENCE (centered on the sampled frame of
            # reference).
            Xbound, Ybound, Zbound = worldToIntrinsic(
                R=spatialRefQ, xWorld=Xbound, yWorld=Ybound, zWorld=Zbound)

            # Rounding to the nearest image position integer
            Xbound = np.round(Xbound).astype(int)
            Ybound = np.round(Ybound).astype(int)
            Zbound = np.round(Zbound).astype(int)

            sizeQ = np.array([Xbound[1] - Xbound[0] + 1, Ybound[1] -
                            Ybound[0] + 1, Zbound[1] - Zbound[0] + 1])

            # Converting back to world positions ion order to correctly define
            # edges of the new box and thus center it onto the full queried
            # reference frame
            Xbound, Ybound, Zbound = intrinsicToWorld(R=spatialRefQ,
                                                    xIntrinsic=Xbound,
                                                    yIntrinsic=Ybound,
                                                    zIntrinsic=Zbound)

            newLowLimitsQ[0] = Xbound[0] - resQ[0]/2
            newLowLimitsQ[1] = Ybound[0] - resQ[1]/2
            newLowLimitsQ[2] = Zbound[0] - resQ[2]/2

            spatialRefQ = imref3d(imageSize=sizeQ, 
                                pixelExtentInWorldX=resQ[0],
                                pixelExtentInWorldY=resQ[1],
                                pixelExtentInWorldZ=resQ[2])

            spatialRefQ.XWorldLimits -= spatialRefQ.XWorldLimits[0] - \
                newLowLimitsQ[0]
            spatialRefQ.YWorldLimits -= spatialRefQ.YWorldLimits[0] - \
                newLowLimitsQ[1]
            spatialRefQ.ZWorldLimits -= spatialRefQ.ZWorldLimits[0] - \
                newLowLimitsQ[2]

        # CREATING QUERIED XYZ POINTS
        Xq = np.arange(sizeQ[0])
        Yq = np.arange(sizeQ[1])
        Zq = np.arange(sizeQ[2])
        Xq, Yq, Zq = np.meshgrid(Xq, Yq, Zq, indexing='ij')
        Xq, Yq, Zq = intrinsicToWorld(
            R=spatialRefQ, xIntrinsic=Xq, yIntrinsic=Yq, zIntrinsic=Zq)

        # CONVERTING QUERIED XZY POINTS TO INTRINSIC COORDINATES IN THE SAMPLED
        # REFERENCE FRAME
        Xq, Yq, Zq = worldToIntrinsic(
            R=spatialRefS, xWorld=Xq, yWorld=Yq, zWorld=Zq)

        # INTERPOLATING VOLUME
        data = interp3(V=volObjS.data, Xq=Xq, Yq=Yq, Zq=Zq, method=interpMet)
        volObjQ = ImageVolumeObj(data=data, spatialRef=spatialRefQ)

        # ROUNDING
        if image_type == "image":
            # Grey level rounding for "image" type
            if roundVal is not None and (type(roundVal) is int or type(roundVal) is float):
                # DELETE NEXT LINE WHEN THE RADIOMICS PARAMETER OPTIONS OF
                # interp.glRound ARE FIXED
                roundVal = (-np.log10(roundVal)).astype(int)
                volObjQ.data = np.around(volObjQ.data, decimals=roundVal)
        else:
            volObjQ.data[volObjQ.data >= roundVal] = 1.0
            volObjQ.data[volObjQ.data < roundVal] = 0.0

    except Exception as e:
        if MEDimage.__dict__['scaleName'] != '':
            message = f"\n PROBLEM WITH PRE-PROCESSING OF TEXTURE FEATURES:\n {e}"
            _logger.error(message)

            MEDimage.Params['radiomics']['image'].update(
                {( MEDimage.__dict__['scaleName'] ): 'ERROR_PROCESSING'})
            
        else:
            message = f"\n PROBLEM WITH PRE-PROCESSING OF TEXTURE FEATURES:\n {e}"
            _logger.error(message)

            MEDimage.Params['radiomics']['image'].update(
                {('scale'+(str(MEDimage.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return volObjQ
