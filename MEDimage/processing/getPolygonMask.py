#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from utils.imref import worldToIntrinsic
from utils.inpolygon import inpolygon
from utils.mode import mode


def getPolygonMask(ROI_XYZ, spatialRef, orientation) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box 
    in all dimensions.

    Args:
        ROI_XYZ (ndarray): array of (x,y,z) triplets defining a contour in the 
            Patient-Based Coordinate System extracted from DICOM RTstruct.
        spatialRef (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        orientation (str): Imaging data orientation (axial, sagittal or coronal).

    Returns:
        array: 3D array of 1's and 0's defining the ROI mask.

    """

    # COMPUTING MASK
    sz = spatialRef.ImageSize.copy()
    ROImask = np.zeros(sz)
    # X,Y,Z in intrinsic image coordinates
    X, Y, Z = worldToIntrinsic(R=spatialRef, 
                               xWorld=ROI_XYZ[:, 0],
                               yWorld=ROI_XYZ[:, 1],
                               zWorld=ROI_XYZ[:, 2])

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
    closedContours = np.unique(ROI_XYZ[:, 3])
    xq = np.arange(sz[0])
    yq = np.arange(sz[1])
    xq, yq = np.meshgrid(xq, yq)

    for cc in np.arange(len(closedContours)):
        ind = ROI_XYZ[:, 3] == closedContours[cc]
        # Taking the mode, just in case. But normally, numel(unique(K(ind)))
        # should evaluate to 1, as closed contours are meant to be defined on
        # a given slice
        select_slice = mode(K[ind]).astype(int)
        inpoly = inpolygon(xq=xq, yq=yq, xv=points[ind, a], yv=points[ind, b])
        ROImask[:, :, select_slice] = np.logical_or(
            ROImask[:, :, select_slice], inpoly)

    return ROImask
