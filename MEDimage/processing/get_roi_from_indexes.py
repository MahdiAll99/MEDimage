#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from typing import Tuple

import numpy as np

from MEDimage import MEDimage

from ..processing.compute_box import compute_box
from ..utils.image_volume_obj import image_volume_obj
from ..utils.parse_contour_string import parse_contour_string
from .get_sep_roi_names import get_sep_roi_names


def get_roi_from_indexes(
        MEDimg: MEDimage, 
        name_roi: str, 
        box_string: str
    ) -> Tuple[image_volume_obj, image_volume_obj]:
    """Extracts the ROI box (+ smallest box containing the region of interest)
    and associated mask from the indexes saved in 'MEDimage' file.
    
    Args:
        MEDimage (MEDimage): The MEDimage class object.
        name_roi (str): name of the ROI since the a volume can have multuiple
            ROIs.
        box_string (str): Specifies the size if the box containing the ROI

            - 'full': Full imaging data as output.
            - 'box': computes the smallest bounding box.
            - Ex: 'box10': 10 voxels in all three dimensions are added to \
                the smallest bounding box. The number after 'box' defines the \
                number of voxels to add.
            - Ex: '2box': Computes the smallest box and outputs double its \
                size. The number before 'box' defines the multiplication in \
                size.

    Returns:
        2-element tuple containing

        - ndarray: vol_obj, 3D array of imaging data defining the smallest box \
            containing the region of interest.
        - ndarray: roi_obj, 3D array of 1's and 0's defining the ROI in ROIbox.
    """
    # This takes care of the "Volume resection" step
    # as well using the argument "box". No fourth
    # argument means 'interp' by default.

    # PARSING OF ARGUMENTS
    try:
        name_structure_set = []
        delimiters = ["\+", "\-"]
        n_contour_data = len(MEDimg.scan.ROI.indexes)

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
                name_temp = MEDimg.scan.ROI.get_roi_name(key=j)
                if name_temp == name_roi[i]:
                    if name_structure_set:
                        # FOR DICOM + RTSTRUCT
                        name_set_temp = MEDimg.scan.ROI.get_name_set(key=j)
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

        contour_number, operations = parse_contour_string(contour_string)

        # INTIALIZATIONS
        if type(contour_number) is int:
            n_contour = 1
            contour_number = [contour_number]
        else:
            n_contour = len(contour_number)

        # Note: sData is a nested dictionary not an object
        spatial_ref = MEDimg.scan.volume.spatialRef
        vol = MEDimg.scan.volume.data.astype(np.float32)

        # APPLYING OPERATIONS ON ALL MASKS
        roi = MEDimg.scan.get_indexes_by_roi_name(name_roi[0])
        for c in np.arange(start=1, stop=n_contour):
            if operations[c-1] == "+":
                roi += MEDimg.scan.get_indexes_by_roi_name(name_roi[c])
            elif operations[c-1] == "-":
                roi -= MEDimg.scan.get_indexes_by_roi_name(name_roi[c])
            else:
                raise ValueError("Unknown operation on ROI.")

            roi[roi >= 1.0] = 1.0
            roi[roi < 1.0] = 0.0

        # COMPUTING THE BOUNDING BOX
        vol, roi, new_spatial_ref = compute_box(vol=vol, roi=roi,
                                            spatial_ref=spatial_ref,
                                            box_string=box_string)

        # ARRANGE OUTPUT
        vol_obj = image_volume_obj(data=vol, spatial_ref=new_spatial_ref)
        roi_obj = image_volume_obj(data=roi, spatial_ref=new_spatial_ref)

    except Exception as e:
        message = f"\n PROBLEM WITH PRE-PROCESSING OF FEATURES IN get_roi_from_indexes():\n {e}"
        logging.error(message)
        print(message)

        MEDimg.radiomics.image.update(
            {('scale'+(str(MEDimg.params.process.scale_non_text[0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return vol_obj, roi_obj