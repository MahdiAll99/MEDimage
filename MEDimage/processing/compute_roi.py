#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ..utils.imref import imref3d
from ..utils.interp3 import interp3

from ..processing.find_spacing import find_spacing
from ..processing.get_polygon_mask import get_polygon_mask


def compute_roi(roi_xyz: np.ndarray,
                spatial_ref: imref3d,
                orientation: str,
                scan_type: str,
                interp=False) -> np.ndarray:
    """Computes the ROI (Region of interest) mask using the XYZ coordinates.

    Note:
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
        --> The 'interp' should be used only if tested and verified. False
                is currently the default in get_roi.py

    Args:
        roi_xyz (ndarray): array of (x,y,z) triplets defining a contour in the Patient-Based
                           Coordinate System extracted from DICOM RTstruct.
        spatial_ref (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        orientation (str): Imaging data orientation (axial, sagittal or coronal).
        scan_type (str): Imaging modality (MRscan, CTscan...).
        interp (bool): Specifies if we need to use an interpolation
                       process prior to "get_polygon_mask()" in the slice axis direction.
                       - True: Interpolation is performed in the slice axis dimensions.
                               To be further tested, thus please use with caution (True is safer).
                       - False (default): No interpolation. This can definitely be safe
                                          when the RTstruct has been saved specifically for the volume of
                                          interest.
    Returns:
        ndarray: 3D array of 1's and 0's defining the ROI mask.

    Todo:
        * USING INTERPOLATION --> THIS PART NEEDS TO BE FURTHER TESTED.
        * Consider changing to if statement. Changing interp variable here
        will change the interp variable everywhere
    """

    while interp:
        # Initialization
        if orientation == "Axial":
            dim_ijk = 2
            dim_xyz = 2
            direction = "Z"
            # Only the resolution in 'Z' will be changed
            res_xyz = np.array([spatial_ref.PixelExtentInWorldX,
                               spatial_ref.PixelExtentInWorldY, 0.0])
        elif orientation == "Sagittal":
            dim_ijk = 0
            dim_xyz = 1
            direction = "Y"
            # Only the resolution in 'Y' will be changed
            res_xyz = np.array([spatial_ref.PixelExtentInWorldX, 0.0,
                               spatial_ref.PixelExtentInWorldZ])
        elif orientation == "Coronal":
            dim_ijk = 1
            dim_xyz = 0
            direction = "X"
            # Only the resolution in 'X' will be changed
            res_xyz = np.array([0.0, spatial_ref.PixelExtentInWorldY,
                               spatial_ref.PixelExtentInWorldZ])
        else:
            raise ValueError(
                "Provided orientation is not one of \"Axial\", \"Sagittal\", \"Coronal\".")

        # Creating new imref3d object for sample points (with slice dimension
        # similar to original volume
        # where RTstruct was created)
        # Slice spacing in mm
        slice_spacing = find_spacing(
            roi_xyz[:, dim_ijk], scan_type).astype(np.float32)

        # Only one slice found in the function "find_spacing" on the above line.
        # We thus must set "slice_spacing" to the slice spacing of the queried
        # volume, and no interpolation will be performed.
        if slice_spacing is None:
            slice_spacing = spatial_ref.PixelExtendInWorld(axis=direction)

        new_size = round(spatial_ref.ImageExtentInWorld(
            axis=direction) / slice_spacing)
        res_xyz[dim_xyz] = slice_spacing
        s_z = spatial_ref.ImageSize.copy()
        s_z[dim_ijk] = new_size

        xWorldLimits = spatial_ref.XWorldLimits.copy()
        yWorldLimits = spatial_ref.YWorldLimits.copy()
        zWorldLimits = spatial_ref.ZWorldLimits.copy()

        new_spatial_ref = imref3d(imageSize=s_z, 
                                pixelExtentInWorldX=res_xyz[0],
                                pixelExtentInWorldY=res_xyz[1],
                                pixelExtentInWorldZ=res_xyz[2],
                                xWorldLimits=xWorldLimits,
                                yWorldLimits=yWorldLimits,
                                zWorldLimits=zWorldLimits)

        diff = (new_spatial_ref.ImageExtentInWorld(axis=direction) -
                spatial_ref.ImageExtentInWorld(axis=direction))

        if np.abs(diff) >= 0.01:
            # Sampled and queried volume are considered "different".
            new_limit = spatial_ref.WorldLimits(axis=direction)[0] - diff / 2.0

            # Sampled volume is now centered on queried volume.
            new_spatial_ref.WorldLimits(axis=direction, newValue=(new_spatial_ref.WorldLimits(axis=direction) -
                                                                (new_spatial_ref.WorldLimits(axis=direction)[0] - 
                                                                 new_limit)))
        else:
            # Less than a 0.01 mm, sampled and queried volume are considered
            # to be the same. At this point,
            # spatial_ref and new_spatial_ref may have differed due to data
            # manipulation, so we simply compute
            # the ROI mask with spatial_ref (i.e. simply using "poly2mask.m"),
            # without performing interpolation.
            interp = False
            break  # Getting out of the "while" statement

        V = get_polygon_mask(roi_xyz, new_spatial_ref, orientation)

        # Getting query points (x_q,y_q,z_q) of output roi_mask
        sz_q = spatial_ref.ImageSize
        x_qi = np.arange(sz_q[0])
        y_qi = np.arange(sz_q[1])
        z_qi = np.arange(sz_q[2])
        x_qi, y_qi, z_qi = np.meshgrid(x_qi, y_qi, z_qi, indexing='ij')

        # Getting queried mask
        v_q = interp3(V=V, x_q=x_qi, y_q=y_qi, z_q=z_qi, method="cubic")
        roi_mask = v_q
        roi_mask[v_q < 0.5] = 0
        roi_mask[v_q >= 0.5] = 1

        # Getting out of the "while" statement
        interp = False

    # SIMPLY USING "poly2mask.m" or "inpolygon.m". "inpolygon.m" is slower, but
    # apparently more accurate.
    if not interp:
        # Using the inpolygon.m function. To be further tested.
        roi_mask = get_polygon_mask(roi_xyz, spatial_ref, orientation)

    return roi_mask
