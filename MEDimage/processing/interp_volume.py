#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
import logging

import numpy as np
from ..utils.image_volume_obj import image_volume_obj
from ..utils.imref import imref3d, intrinsicToWorld, worldToIntrinsic
from ..utils.interp3 import interp3

from ..processing.compute_box import compute_box

_logger = logging.getLogger(__name__)


def interp_volume(MEDimage, 
                volObjS, 
                voxDim=None, 
                interpMet=None, 
                roundVal=None,
                image_type=None, 
                roiObjS=None, 
                box_string=None) -> image_volume_obj:
    """3D voxel interpolation on the input volume.

    Args:
        MEDimage (object): The MEDimage class object.
        volObjS (image_volume_obj): Imaging  that will be interpolated.
        voxDim (array): Array of the voxel dimension. The following format is used 
            [Xin,Yin,Zslice], where Xin and Yin are the X (left to right) and 
            Y (bottom to top) IN-PLANE resolutions, and Zslice is the slice spacing,
            NO MATTER THE ORIENTATION OF THE VOLUME (i.e. axial , sagittal, coronal).
        interpMet (str): {nearest, linear, spline, cubic} optional, Interpolation method
        roundVal (float): Rounding value. Must be between 0 and 1 for ROI interpolation
            and to a power of 10 for Image interpolation.
        image_type (str): 'image' for imaging data interpolation and 'roi' for ROI mask
            data interpolation.
        roiObjS (image_volume_obj): Mask data, will be used to compute a new specific box 
            and the new imref3d object for the imaging data.
        box_string (str): Specifies the size if the box containing the ROI
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
            two_d = True
        else:
            two_d = False

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

        if roiObjS is None or box_string is None:
            useBox = False
        else:
            useBox = True

        # --> QUERIED POINTS: NEW INTERPOLATED VOLUME: "q" or "Q".
        # --> SAMPLED POINTS: ORIGINAL VOLUME: "s" or "S".
        # --> Always using XYZ coordinates (unless specifically noted),
        #     not MATLAB IJK, so beware!

        # INITIALIZATION
        res_q = voxDim
        if two_d:
            # If 2D, the resolution of the slice dimension of he queried volume is
            # set to the same as the sampled volume.
            res_q = np.concatenate((res_q, volObjS.spatial_ref.PixelExtentInWorldZ))

        res_s = np.array([volObjS.spatial_ref.PixelExtentInWorldX,
                        volObjS.spatial_ref.PixelExtentInWorldY,
                        volObjS.spatial_ref.PixelExtentInWorldZ])

        if np.array_equal(res_s, res_q):
            return deepcopy(volObjS)

        spatial_ref_s = volObjS.spatial_ref
        extent_s = np.array([spatial_ref_s.ImageExtentInWorldX,
                            spatial_ref_s.ImageExtentInWorldY,
                            spatial_ref_s.ImageExtentInWorldZ])
        low_limits_s = np.array([spatial_ref_s.XWorldLimits[0],
                            spatial_ref_s.YWorldLimits[0],
                            spatial_ref_s.ZWorldLimits[0]])

        # CREATING QUERIED "imref3d" OBJECT CENTERED ON SAMPLED VOLUME

        # Switching to IJK (matlab) reference frame for "imref3d" computation.
        # Putting a "ceil", according to IBSI standards. This is safer than "round".
        size_q = np.ceil(np.around(np.divide(extent_s, res_q),
                                decimals=3)).astype(int).tolist()

        if two_d:
            # If 2D, forcing the size of the queried volume in the slice dimension
            # to be the same as the sample volume.
            size_q[2] = volObjS.spatial_ref.ImageSize[2]

        spatial_ref_q = imref3d(imageSize=size_q, 
                            pixelExtentInWorldX=res_q[0],
                            pixelExtentInWorldY=res_q[1],
                            pixelExtentInWorldZ=res_q[2])

        extent_q = np.array([spatial_ref_q.ImageExtentInWorldX,
                            spatial_ref_q.ImageExtentInWorldY,
                            spatial_ref_q.ImageExtentInWorldZ])
        low_limits_q = np.array([spatial_ref_q.XWorldLimits[0],
                            spatial_ref_q.YWorldLimits[0],
                            spatial_ref_q.ZWorldLimits[0]])
        diff = extent_q - extent_s
        new_low_limits_q = low_limits_s - diff/2
        spatial_ref_q.XWorldLimits = spatial_ref_q.XWorldLimits - \
            (low_limits_q[0] - new_low_limits_q[0])
        spatial_ref_q.YWorldLimits = spatial_ref_q.YWorldLimits - \
            (low_limits_q[1] - new_low_limits_q[1])
        spatial_ref_q.ZWorldLimits = spatial_ref_q.ZWorldLimits - \
            (low_limits_q[2] - new_low_limits_q[2])

        # REDUCE THE SIZE OF THE VOLUME PRIOR TO INTERPOLATION
        # TODO check that compute_box vol and roi are intended to be the same!
        if useBox:
            _, _, tempSpatialRef = compute_box(
                vol=roiObjS.data, roi=roiObjS.data, spatial_ref=volObjS.spatial_ref,
                box_string=box_string)

            size_temp = tempSpatialRef.ImageSize

            # Getting world boundaries (center of voxels) of the new box
            x_bound, y_bound, z_bound = intrinsicToWorld(R=tempSpatialRef,
                                                    xIntrinsic=np.array(
                                                        [0.0, size_temp[0]-1.0]),
                                                    yIntrinsic=np.array(
                                                        [0.0, size_temp[1]-1.0]),
                                                    zIntrinsic=np.array([0.0, size_temp[2]-1.0]))

            # Getting the image positions of the boundaries of the new box, IN THE
            # FULL QUERIED FRAME OF REFERENCE (centered on the sampled frame of
            # reference).
            x_bound, y_bound, z_bound = worldToIntrinsic(
                R=spatial_ref_q, xWorld=x_bound, yWorld=y_bound, zWorld=z_bound)

            # Rounding to the nearest image position integer
            x_bound = np.round(x_bound).astype(int)
            y_bound = np.round(y_bound).astype(int)
            z_bound = np.round(z_bound).astype(int)

            size_q = np.array([x_bound[1] - x_bound[0] + 1, y_bound[1] -
                            y_bound[0] + 1, z_bound[1] - z_bound[0] + 1])

            # Converting back to world positions ion order to correctly define
            # edges of the new box and thus center it onto the full queried
            # reference frame
            x_bound, y_bound, z_bound = intrinsicToWorld(R=spatial_ref_q,
                                                    xIntrinsic=x_bound,
                                                    yIntrinsic=y_bound,
                                                    zIntrinsic=z_bound)

            new_low_limits_q[0] = x_bound[0] - res_q[0]/2
            new_low_limits_q[1] = y_bound[0] - res_q[1]/2
            new_low_limits_q[2] = z_bound[0] - res_q[2]/2

            spatial_ref_q = imref3d(imageSize=size_q, 
                                pixelExtentInWorldX=res_q[0],
                                pixelExtentInWorldY=res_q[1],
                                pixelExtentInWorldZ=res_q[2])

            spatial_ref_q.XWorldLimits -= spatial_ref_q.XWorldLimits[0] - \
                new_low_limits_q[0]
            spatial_ref_q.YWorldLimits -= spatial_ref_q.YWorldLimits[0] - \
                new_low_limits_q[1]
            spatial_ref_q.ZWorldLimits -= spatial_ref_q.ZWorldLimits[0] - \
                new_low_limits_q[2]

        # CREATING QUERIED XYZ POINTS
        x_q = np.arange(size_q[0])
        y_q = np.arange(size_q[1])
        z_q = np.arange(size_q[2])
        x_q, y_q, z_q = np.meshgrid(x_q, y_q, z_q, indexing='ij')
        x_q, y_q, z_q = intrinsicToWorld(
            R=spatial_ref_q, xIntrinsic=x_q, yIntrinsic=y_q, zIntrinsic=z_q)

        # CONVERTING QUERIED XZY POINTS TO INTRINSIC COORDINATES IN THE SAMPLED
        # REFERENCE FRAME
        x_q, y_q, z_q = worldToIntrinsic(
            R=spatial_ref_s, xWorld=x_q, yWorld=y_q, zWorld=z_q)

        # INTERPOLATING VOLUME
        data = interp3(V=volObjS.data, x_q=x_q, y_q=y_q, z_q=z_q, method=interpMet)
        vol_obj_q = image_volume_obj(data=data, spatial_ref=spatial_ref_q)

        # ROUNDING
        if image_type == "image":
            # Grey level rounding for "image" type
            if roundVal is not None and (type(roundVal) is int or type(roundVal) is float):
                # DELETE NEXT LINE WHEN THE RADIOMICS PARAMETER OPTIONS OF
                # interp.glRound ARE FIXED
                roundVal = (-np.log10(roundVal)).astype(int)
                vol_obj_q.data = np.around(vol_obj_q.data, decimals=roundVal)
        else:
            vol_obj_q.data[vol_obj_q.data >= roundVal] = 1.0
            vol_obj_q.data[vol_obj_q.data < roundVal] = 0.0

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

    return vol_obj_q
