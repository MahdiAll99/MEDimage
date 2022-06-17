#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict

import numpy as np

from .compute_bounding_box import compute_bounding_box


def get_diag_features(vol_obj, roi_obj_int, roi_obj_morph, im_type) -> Dict:
    """Computes diagnostic features 
    
    The diagnostic features help identify issues with 
    the implementation of the image processing sequence.

    Args:
        vol_obj (image_volume_obj): Imaging data.
        roi_obj_int (image_volume_obj): Intensity mask data.
        roi_obj_morph (image_volume_obj): Morphological mask data.
        im_type (str): Image processing step.
            
            - 'reSeg': Computes Diagnostic features right after the
                re-segmentaion step.
            
            - 'interp' or any other arg: Computes Diagnostic features 
                for any processing step other than re-segmentation.

    Returns:
        Dict: Dictionnary containing the computed features.

    """
    diag = {}

    #  FOR THE IMAGE

    if im_type != 'reSeg':
        # Image dimension x
        diag.update({'image_' + im_type + '_dimX':
                     vol_obj.spatial_ref.ImageSize[0]})

        # Image dimension y
        diag.update({'image_' + im_type + '_dimY':
                     vol_obj.spatial_ref.ImageSize[1]})

        # Image dimension z
        diag.update({'image_' + im_type + '_dimz':
                     vol_obj.spatial_ref.ImageSize[2]})

        # Voxel dimension x
        diag.update({'image_' + im_type + '_voxDimX':
                     vol_obj.spatial_ref.PixelExtentInWorldX})

        # Voxel dimension y
        diag.update({'image_' + im_type + '_voxDimY':
                     vol_obj.spatial_ref.PixelExtentInWorldY})

        # Voxel dimension z
        diag.update({'image_' + im_type + '_voxDimZ':
                     vol_obj.spatial_ref.PixelExtentInWorldZ})

        # Mean intensity
        diag.update({'image_' + im_type + '_meanInt': np.mean(vol_obj.data)})

        # Minimum intensity
        diag.update({'image_' + im_type + '_minInt': np.min(vol_obj.data)})

        # Maximum intensity
        diag.update({'image_' + im_type + '_maxInt': np.max(vol_obj.data)})

    # FOR THE ROI
    box_bound_int = compute_bounding_box(roi_obj_int.data)
    box_bound_morph = compute_bounding_box(roi_obj_morph.data)

    x_gl_int = vol_obj.data[roi_obj_int.data == 1]
    x_gl_morph = vol_obj.data[roi_obj_morph.data == 1]

    # Map dimension x
    diag.update({'roi_' + im_type + '_Int_dimX':
                 roi_obj_int.spatial_ref.ImageSize[0]})

    # Map dimension y
    diag.update({'roi_' + im_type + '_Int_dimY':
                 roi_obj_int.spatial_ref.ImageSize[1]})

    # Map dimension z
    diag.update({'roi_' + im_type + '_Int_dimZ':
                 roi_obj_int.spatial_ref.ImageSize[2]})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Int_boxBoundDimX':
                 box_bound_int[0, 1] - box_bound_int[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Int_boxBoundDimY':
                 box_bound_int[1, 1] - box_bound_int[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Int_boxBoundDimZ':
                 box_bound_int[2, 1] - box_bound_int[2, 0] + 1})

    # Bounding box dimension x
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimX':
                 box_bound_morph[0, 1] - box_bound_morph[0, 0] + 1})

    # Bounding box dimension y
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimY':
                 box_bound_morph[1, 1] - box_bound_morph[1, 0] + 1})

    # Bounding box dimension z
    diag.update({'roi_' + im_type + '_Morph_boxBoundDimZ':
                 box_bound_morph[2, 1] - box_bound_morph[2, 0] + 1})

    # Voxel number
    diag.update({'roi_' + im_type + '_Int_voxNumb': np.size(x_gl_int)})

    # Voxel number
    diag.update({'roi_' + im_type + '_Morph_voxNumb': np.size(x_gl_morph)})

    # Mean intensity
    diag.update({'roi_' + im_type + '_meanInt': np.mean(x_gl_int)})

    # Minimum intensity
    diag.update({'roi_' + im_type + '_minInt': np.min(x_gl_int)})

    # Maximum intensity
    diag.update({'roi_' + im_type + '_maxInt': np.max(x_gl_int)})

    return diag