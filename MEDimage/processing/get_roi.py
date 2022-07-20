#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from typing import Union

import numpy as np
from MEDimage import MEDimage

from ..processing.compute_box import compute_box
from ..processing.compute_roi import compute_roi
from ..utils.image_volume_obj import image_volume_obj
from ..utils.parse_contour_string import parse_contour_string
from .get_sep_roi_names import get_sep_roi_names

_logger = logging.getLogger(__name__)

def get_roi(MEDimage: MEDimage,
            name_roi: str,
            box_string: str,
            interp=False) -> Union[image_volume_obj,
                                   image_volume_obj]:
    """Computes the ROI box (box containing the region of interest)
    and associated mask from MEDimage object.

    Args:
        MEDimage (MEDimage): The MEDimage class object.
        name_roi (str): name of the ROI since the a volume can have multuiple ROIs.
        box_string (str): Specifies the size if the box containing the ROI
                          - 'full': Full imaging data as output.
                          - 'box' computes the smallest bounding box.
                          - Ex: 'box10': 10 voxels in all three dimensions are added to
                            the smallest bounding box. The number after 'box' defines the
                            number of voxels to add.
                          - Ex: '2box': Computes the smallest box and outputs double its
                            size. The number before 'box' defines the multiplication in size.
        interp (bool): True if we need to use an interpolation for box computation.

    Returns:
        image_volume_obj: 3D array of imaging data defining box containing the ROI.
            vol.data is the 3D array, vol.spatialRef is its associated imref3d object.
        image_volume_obj: 3D array of 1's and 0's defining the ROI.
            roi.data is the 3D array, roi.spatialRef is its associated imref3d object.
    """
    # PARSING OF ARGUMENTS
    try:
        name_structure_set = []
        delimiters = ["\+", "\-"]
        n_contour_data = len(MEDimage.scan.ROI.indexes)

        name_roi, vect_plus_minus = get_sep_roi_names(name_roi, delimiters)
        contour_number = np.zeros(len(name_roi))

        if name_structure_set is None:
            name_structure_set = []

        if name_structure_set:
            name_structure_set, _ = get_sep_roi_names(name_structure_set, delimiters)
            if len(name_roi) != len(name_structure_set):
                raise ValueError(
                    "The numbers of defined ROI names and Structure Set names are not the same")

        for i in range(0, len(name_roi)):
            for j in range(0, n_contour_data):
                name_temp = MEDimage.scan.ROI.get_roi_name(key=j)
                if name_temp == name_roi[i]:
                    if name_structure_set:
                        # FOR DICOM + RTSTRUCT
                        name_set_temp = MEDimage.scan.ROI.get_name_set(key=j)
                        if name_set_temp == name_structure_set[i]:
                            contour_number[i] = j
                            break
                    else:
                        contour_number[i] = j
                        break

        n_roi = np.size(contour_number)
        # contour_string IS FOR EXAMPLE '3' or '1-3+2'
        contour_string = str(contour_number[0].astype(int))

        for i in range(1, n_roi):
            if vect_plus_minus[i-1] == 1:
                sign = '+'
            elif vect_plus_minus[i-1] == -1:
                sign = '-'
            contour_string = contour_string + sign + \
                str(contour_number[i].astype(int))

        if not (box_string == "full" or "box" in box_string):
            raise ValueError(
                "The third argument must either be \"full\" or contain the word \"box\".")

        if type(interp) != bool:
            raise ValueError(
                "If present (i.e. it is optional), the fourth argument must be bool")

        contour_number, operations = parse_contour_string(contour_string)

        # INTIALIZATIONS
        if type(contour_number) is int:
            n_contour = 1
            contour_number = [contour_number]
        else:
            n_contour = len(contour_number)

        roi_mask_list = []
        if MEDimage.type not in ["PTscan", "CTscan", "MRscan", "ADCscan"]:
            raise ValueError("Unknown scan type.")

        spatial_ref = MEDimage.scan.volume.spatialRef
        vol = MEDimage.scan.volume.data.astype(np.float32)

        # COMPUTING ALL MASKS
        for c in np.arange(start=0, stop=n_contour):
            contour = contour_number[c]
            # GETTING THE XYZ POINTS FROM MEDimage
            roi_xyz = MEDimage.scan.ROI.get_indexes(key=contour).copy()

            # APPLYING ROTATION TO XYZ POINTS (if necessary --> MRscan)
            if hasattr(MEDimage.scan.volume, 'scanRot') and MEDimage.scan.volume.scanRot is not None:
                roi_xyz[:, [0, 1, 2]] = np.transpose(
                    MEDimage.scan.volume.scanRot @ np.transpose(roi_xyz[:, [0, 1, 2]]))

            # APPLYING TRANSLATION IF SIMULATION STRUCTURE AS INPUT
            # (software STAMP utility)
            if hasattr(MEDimage.scan.volume, 'transScanToModel'):
                translation = MEDimage.scan.volume.transScanToModel
                roi_xyz[:, 0] += translation[0]
                roi_xyz[:, 1] += translation[1]
                roi_xyz[:, 2] += translation[2]

            # COMPUTING THE ROI MASK
            # Problem here in compute_roi.m: If the volume is a full-body CT and the
            # slice interpolation process occurs, a lot of RAM will be used.
            # One solution could be to a priori compute the bounding box before
            # computing the ROI (using XYZ points). But we still want the user to
            # be able to fully use the "box" argument, so we are fourré...TO SOLVE!
            roi_mask_list += [compute_roi(roi_xyz=roi_xyz, 
                                        spatial_ref=spatial_ref,
                                        orientation=MEDimage.scan.orientation,
                                        scan_type=MEDimage.type,
                                        interp=interp).astype(np.float32)]

        # APPLYING OPERATIONS ON ALL MASKS
        roi = roi_mask_list[0]
        for c in np.arange(start=1, stop=n_contour):
            if operations[c-1] == "+":
                roi += roi_mask_list[c]
            elif operations[c-1] == "-":
                roi -= roi_mask_list[c]
            else:
                raise ValueError("Unknown operation on ROI.")

            roi[roi >= 1.0] = 1.0
            roi[roi < 1.0] = 0.0

        # COMPUTING THE BOUNDING BOX
        vol, roi, new_spatial_ref = compute_box(vol=vol, 
                                            roi=roi,
                                            spatial_ref=spatial_ref,
                                            box_string=box_string)

        # ARRANGE OUTPUT
        vol_obj = image_volume_obj(data=vol, spatial_ref=new_spatial_ref)
        roi_obj = image_volume_obj(data=roi, spatial_ref=new_spatial_ref)

    except Exception as e:
        message = f"\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN get_roi(): \n {e}"
        _logger.error(message)

        MEDimage.Params['radiomics']['image'].update(
            {('scale'+(str(MEDimage.Params['scaleNonText'][0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return vol_obj, roi_obj
