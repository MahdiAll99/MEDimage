#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

import computeBoundingBox


def getDiagFeatures(volObj, roiObj_Int, roiObj_Morph, im_type) -> Dict:
    """Computes diagnostic features 
    
    The diagnostic features help identify issues with 
    the implementation of the image processing sequence

    Args:
        volObj (ImageVolumeObj): Imagign data.
        roiObj_Int (ImageVolumeObj): Mask data.
        roiObj_Morph (ImageVolumeObj): Morphological mask data.
        im_type (str): Image processing step.
        --> Ex: - 'reSeg': Computes Diagnostic features right after the
                    re-segmentaion step.
                - 'interp' or any other arg: Computes Diagnostic features 
                    for any processing step other than re-segmentation

    Returns:
        Dict: Dictionnary containing the computed features.

    """
    diag = {}

    #  FOR THE IMAGE

    if im_type != 'reSeg':
        # Image dimension x
        diag.update({'image_' + im_type + '_dimX':
                     volObj.spatialRef.ImageSize[0]})

        # Image dimension y
        diag.update({'image_' + im_type + '_dimY':
                     volObj.spatialRef.ImageSize[1]})

        # Image dimension z
        diag.update({'image_' + im_type + '_dimz':
                     volObj.spatialRef.ImageSize[2]})

        # Voxel dimension x
        diag.update({'image_' + im_type + '_voxDimX':
                     volObj.spatialRef.PixelExtentInWorldX})

        # Voxel dimension y
        diag.update({'image_' + im_type + '_voxDimY':
                     volObj.spatialRef.PixelExtentInWorldY})

        # Voxel dimension z
        diag.update({'image_' + im_type + '_voxDimZ':
                     volObj.spatialRef.PixelExtentInWorldZ})

        # Mean intensity
        diag.update({'image_' + im_type + '_meanInt': np.mean(volObj.data)})

        # Minimum intensity
        diag.update({'image_' + im_type + '_minInt': np.min(volObj.data)})

        # Maximum intensity
        diag.update({'image_' + im_type + '_maxInt': np.max(volObj.data)})

    # FOR THE ROI
    boxBound_Int = computeBoundingBox(
        roiObj_Int.data)
    boxBound_Morph = computeBoundingBox(
        roiObj_Morph.data)
    Xgl_Int = volObj.data[roiObj_Int.data == 1]
    Xgl_Morph = volObj.data[roiObj_Morph.data == 1]

    # Map dimension x
    diag.update({'roi_' + im_type + '_Int_dimX':
                 roiObj_Int.spatialRef.ImageSize[0]})

    # Map dimension y
    diag.update({'roi_' + im_type + '_Int_dimY':
                 roiObj_Int.spatialRef.ImageSize[1]})

    # Map dimension z
    diag.update({'roi_' + im_type + '_Int_dimZ':
                 roiObj_Int.spatialRef.ImageSize[2]})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Int_boxBoundDimX':
                 boxBound_Int[0, 1] - boxBound_Int[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Int_boxBoundDimY':
                 boxBound_Int[1, 1] - boxBound_Int[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Int_boxBoundDimZ':
                 boxBound_Int[2, 1] - boxBound_Int[2, 0] + 1})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimX':
                 boxBound_Morph[0, 1] - boxBound_Morph[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimY':
                 boxBound_Morph[1, 1] - boxBound_Morph[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimZ':
                 boxBound_Morph[2, 1] - boxBound_Morph[2, 0] + 1})

    # Voxel number
    diag.update({'roi_' + im_type + '_Int_voxNumb': np.size(Xgl_Int)})

    # Voxel number
    diag.update({'roi_' + im_type + '_Morph_voxNumb': np.size(Xgl_Morph)})

    # Mean intensity
    diag.update({'roi_' + im_type + '_meanInt': np.mean(Xgl_Int)})

    # Minimum intensity
    diag.update({'roi_' + im_type + '_minInt': np.min(Xgl_Int)})

    # Maximum intensity
    diag.update({'roi_' + im_type + '_maxInt': np.max(Xgl_Int)})

    return diag
