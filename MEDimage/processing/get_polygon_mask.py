#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from ..utils.imref import worldToIntrinsic
from ..utils.inpolygon import inpolygon
from ..utils.mode import mode


def get_polygon_mask(roi_xyz, spatial_ref, orientation) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box 
    in all dimensions.

    Args:
        roi_xyz (ndarray): array of (x,y,z) triplets defining a contour in the 
            Patient-Based Coordinate System extracted from DICOM RTstruct.
        spatial_ref (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        orientation (str): Imaging data orientation (axial, sagittal or coronal).

    Returns:
        ndarray: 3D array of 1's and 0's defining the ROI mask.

    """

    # COMPUTING MASK
    sz = spatial_ref.ImageSize.copy()
    roi_mask = np.zeros(sz)
    # X,Y,Z in intrinsic image coordinates
    X, Y, Z = worldToIntrinsic(R=spatial_ref, 
                               xWorld=roi_xyz[:, 0],
                               yWorld=roi_xyz[:, 1],
                               zWorld=roi_xyz[:, 2])

    points = np.transpose(np.vstack((X, Y, Z)))

    if orientation == "Axial":
        a = 0
        b = 1
        c = 2
    elif orientation == "Sagittal":
        a = 1
        b = 2
        c = 0
    elif orientation == "Coronal":
        a = 0
        b = 2
        c = 1
    else:
        raise ValueError(
            "Provided orientation is not one of \"Axial\", \"Sagittal\", \"Coronal\".")

    K = np.round(points[:, c])  # Must assign the points to one slice
    closed_contours = np.unique(roi_xyz[:, 3])
    xq = np.arange(sz[0])
    yq = np.arange(sz[1])
    xq, yq = np.meshgrid(xq, yq)

    for cc in np.arange(len(closed_contours)):
        ind = roi_xyz[:, 3] == closed_contours[cc]
        # Taking the mode, just in case. But normally, numel(unique(K(ind)))
        # should evaluate to 1, as closed contours are meant to be defined on
        # a given slice
        select_slice = mode(K[ind]).astype(int)
        inpoly = inpolygon(xq=xq, yq=yq, xv=points[ind, a], yv=points[ind, b])
        roi_mask[:, :, select_slice] = np.logical_or(
            roi_mask[:, :, select_slice], inpoly)

    return roi_mask
