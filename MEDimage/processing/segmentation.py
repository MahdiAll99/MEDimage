#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import logging
from copy import deepcopy
from typing import List, Sequence, Tuple, Union

import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import center_of_mass

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from ..utils.imref import imref3d, intrinsicToWorld, worldToIntrinsic
from ..utils.inpolygon import inpolygon
from ..utils.interp3 import interp3
from ..utils.mode import mode
from ..utils.parse_contour_string import parse_contour_string
from ..utils.strfind import strfind

_logger = logging.getLogger(__name__)


def get_roi_from_indexes(
        medscan: MEDscan, 
        name_roi: str, 
        box_string: str
    ) -> Tuple[image_volume_obj, image_volume_obj]:
    """Extracts the ROI box (+ smallest box containing the region of interest)
    and associated mask from the indexes saved in ``medscan`` scan.
    
    Args:
        medscan (MEDscan): The MEDscan class object.
        name_roi (str): name of the ROI since the a volume can have multiple
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
        n_contour_data = len(medscan.data.ROI.indexes)

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
                name_temp = medscan.data.ROI.get_roi_name(key=j)
                if name_temp == name_roi[i]:
                    if name_structure_set:
                        # FOR DICOM + RTSTRUCT
                        name_set_temp = medscan.data.ROI.get_name_set(key=j)
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
        spatial_ref = medscan.data.volume.spatialRef
        vol = medscan.data.volume.array.astype(np.float32)

        # APPLYING OPERATIONS ON ALL MASKS
        roi = medscan.data.get_indexes_by_roi_name(name_roi[0])
        for c in np.arange(start=1, stop=n_contour):
            if operations[c-1] == "+":
                roi += medscan.data.get_indexes_by_roi_name(name_roi[c])
            elif operations[c-1] == "-":
                roi -= medscan.data.get_indexes_by_roi_name(name_roi[c])
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

        medscan.radiomics.image.update(
            {('scale'+(str(medscan.params.process.scale_non_text[0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return vol_obj, roi_obj

def get_sep_roi_names(name_roi_in: str,
                      delimiters: List) -> Tuple[List[int],
                                                 np.ndarray]:
    """Seperated ROI names present in the given ROI name. An ROI name can
    have multiple ROI names seperated with curly brackets and delimeters.
    Note:
        Works only for delimiters "+" and "-".
    Args:
        name_roi_in (str): Name of ROIs that will be extracted from the imagign volume. \
                           Separated with curly brackets and delimeters. Ex: '{ED}+{ET}'.
        delimiters (List): List of delimeters of "+" and "-".
    Returns:
        2-element tuple containing 
        
        - List[int]: List of ROI names seperated and excluding curly brackets.
        - ndarray: array of 1's and -1's that defines the regions that will \
                 included and/or excluded in/from the imaging data.
    Examples:
        >>> get_sep_roi_names('{ED}+{ET}', ['+', '-'])
        ['ED', 'ET'], [1]
        >>> get_sep_roi_names('{ED}-{ET}', ['+', '-'])
        ['ED', 'ET'], [-1]
    """
    # EX:
    #name_roi_in = '{GTV-1}'
    #delimiters = ['\\+','\\-']

    # FINDING "+" and "-"
    ind_plus = strfind(string=name_roi_in, pattern=delimiters[0])
    vect_plus = np.ones(len(ind_plus))
    ind_minus = strfind(string=name_roi_in, pattern=delimiters[1])
    vect_minus = np.ones(len(ind_minus)) * -1
    ind = np.argsort(np.hstack((ind_plus, ind_minus)))
    vect_plus_minus = np.hstack((vect_plus, vect_minus))[ind]
    ind = np.hstack((ind_plus, ind_minus))[ind].astype(int)
    n_delim = np.size(vect_plus_minus)

    # MAKING SURE "+" and "-" ARE NOT INSIDE A ROIname
    ind_start = strfind(string=name_roi_in, pattern="{")
    n_roi = len(ind_start)
    ind_stop = strfind(string=name_roi_in, pattern="}")
    ind_keep = np.ones(n_delim, dtype=bool)
    for d in np.arange(n_delim):
        for r in np.arange(n_roi):
             # Thus not indise a ROI name
            if (ind_stop[r] - ind[d]) > 0 and (ind[d] - ind_start[r]) > 0:
                ind_keep[d] = False
                break

    ind = ind[ind_keep]
    vect_plus_minus = vect_plus_minus[ind_keep]

    # PARSING ROI NAMES
    if ind.size == 0:
        # Excluding the "{" and "}" at the start and end of the ROIname
        name_roi_out = [name_roi_in[1:-1]]
    else:
        n_ind = len(ind)
        # Excluding the "{" and "}" at the start and end of the ROIname
        name_roi_out = [name_roi_in[1:(ind[0]-1)]]
        for i in np.arange(start=1, stop=n_ind):
            # Excluding the "{" and "}" at the start and end of the ROIname
            name_roi_out += [name_roi_in[(ind[i-1]+2):(ind[i]-1)]]
        name_roi_out += [name_roi_in[(ind[-1]+2):-1]]

    return name_roi_out, vect_plus_minus
    
def roi_extract(vol: np.ndarray,
                roi: np.ndarray) -> np.ndarray:
    """Replaces volume intensities outside the ROI with NaN.

    Args:
        vol (ndarray): Imaging data.
        roi (ndarray): ROI mask with values of 0's and 1's.

    Returns:
        ndarray: Imaging data with original intensities in the ROI \
            and NaN for intensities outside the ROI.
    """

    vol_re = deepcopy(vol)
    vol_re[roi == 0] = np.nan

    return vol_re

def get_polygon_mask(roi_xyz: np.ndarray,
                     spatial_ref: imref3d) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box
    in all dimensions.

    Args:
        roi_xyz (ndarray): array of (x,y,z) triplets defining a contour in the
                           Patient-Based Coordinate System extracted from DICOM RTstruct.
        spatial_ref (imref3d): imref3d object (same functionality of MATLAB imref3d class).

    Returns:
        ndarray: 3D array of 1's and 0's defining the ROI mask.
    """

    # COMPUTING MASK
    s_z = spatial_ref.ImageSize.copy()
    roi_mask = np.zeros(s_z)
    # X,Y,Z in intrinsic image coordinates
    X, Y, Z = worldToIntrinsic(R=spatial_ref,
                               xWorld=roi_xyz[:, 0],
                               yWorld=roi_xyz[:, 1],
                               zWorld=roi_xyz[:, 2])

    points = np.transpose(np.vstack((X, Y, Z)))

    K = np.round(points[:, 2])  # Must assign the points to one slice
    closed_contours = np.unique(roi_xyz[:, 3])
    x_q = np.arange(s_z[0])
    y_q = np.arange(s_z[1])
    x_q, y_q = np.meshgrid(x_q, y_q)

    for c_c in np.arange(len(closed_contours)):
        ind = roi_xyz[:, 3] == closed_contours[c_c]
        # Taking the mode, just in case. But normally, numel(unique(K(ind)))
        # should evaluate to 1, as closed contours are meant to be defined on
        # a given slice
        select_slice = mode(K[ind]).astype(int)
        inpoly = inpolygon(x_q=x_q, y_q=y_q, x_v=points[ind, 0], y_v=points[ind, 1])
        roi_mask[:, :, select_slice] = np.logical_or(
            roi_mask[:, :, select_slice], inpoly)

    return roi_mask

def voxel_to_spatial(affine: np.ndarray,
                     voxel_pos: list) -> np.array:
    """Convert voxel position into spatial position.

    Args:
        affine (ndarray): Affine matrix.
        voxel_pos (list): A list that correspond to the location in voxel.

    Returns:
        ndarray: A numpy array that correspond to the spatial position in mm.
    """
    m = affine[:3, :3]
    translation = affine[:3, 3]
    return m.dot(voxel_pos) + translation

def spatial_to_voxel(affine: np.ndarray,
                     spatial_pos: list) -> np.array:
    """Convert spatial position into voxel position

    Args:
        affine (ndarray): Affine matrix.
        spatial_pos (list): A list that correspond to the spatial location in mm.

    Returns:
        ndarray: A numpy array that correspond to the position in the voxel.
    """
    affine = np.linalg.inv(affine)
    m = affine[:3, :3]
    translation = affine[:3, 3]
    return m.dot(spatial_pos) + translation

def crop_nifti_box(image: Nifti1Image,
                   roi: Nifti1Image,
                   crop_shape: List[int],
                   center: Union[Sequence[int], None] = None) -> Tuple[Nifti1Image,
                                                                       Nifti1Image]:
    """Crops the Nifti image and ROI.

    Args:
        image (Nifti1Image): Class for the file NIfTI1 format image that will be cropped.
        roi (Nifti1Image): Class for the file NIfTI1 format ROI that will be cropped.
        crop_shape (List[int]): The dimension of the region to crop in term of number of voxel.
        center (Union[Sequence[int], None]): A list that indicate the center of the cropping box
                                             in term of spatial position.

    Returns:
        Tuple[Nifti1Image, Nifti1Image] : Two Nifti images of the cropped image and roi
    """
    assert np.sum(np.array(crop_shape) % 2) == 0, "All elements of crop_shape should be even number."

    image_data = image.get_fdata()
    roi_data = roi.get_fdata()

    radius = [int(x / 2) - 1 for x in crop_shape]
    if center is None:
        center = list(np.array(list(center_of_mass(roi_data))).astype(int))

    center_min = np.floor(center).astype(int)
    center_max = np.ceil(center).astype(int)

    # If center_max and center_min are equal we add 1 to center_max to avoid trouble with crop.
    for i in range(3):
        center_max[i] += 1 if center_max[i] == center_min[i] else 0

    img_shape = image.header['dim'][1:4]

    # Pad the image and the ROI if its necessary
    padding = []
    for rad, cent_min, cent_max, shape in zip(radius, center_min, center_max, img_shape):
        padding.append(
            [abs(min(cent_min - rad, 0)), max(cent_max + rad + 1 - shape, 0)]
        )

    image_data = np.pad(image_data, tuple([tuple(x) for x in padding]))
    roi_data = np.pad(roi_data, tuple([tuple(x) for x in padding]))

    center_min = [center_min[i] + padding[i][0] for i in range(3)]
    center_max = [center_max[i] + padding[i][0] for i in range(3)]

    # Crop the image
    image_data = image_data[center_min[0] - radius[0]:center_max[0] + radius[0] + 1,
                center_min[1] - radius[1]:center_max[1] + radius[1] + 1,
                center_min[2] - radius[2]:center_max[2] + radius[2] + 1]
    roi_data = roi_data[center_min[0] - radius[0]:center_max[0] + radius[0] + 1,
                center_min[1] - radius[1]:center_max[1] + radius[1] + 1,
                center_min[2] - radius[2]:center_max[2] + radius[2] + 1]

    # Update the image and the ROI
    image = Nifti1Image(image_data, affine=image.affine, header=image.header)
    roi = Nifti1Image(roi_data, affine=roi.affine, header=roi.header)

    return image, roi

def crop_box(image_data: np.ndarray,
             roi_data: np.ndarray,
             crop_shape: List[int],
             center: Union[Sequence[int], None] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Crops the imaging data and the ROI mask.

    Args:
        image_data (ndarray): Imaging data that will be cropped.
        roi_data (ndarray): Mask data that will be cropped.
        crop_shape (List[int]): The dimension of the region to crop in term of number of voxel.
        center (Union[Sequence[int], None]): A list that indicate the center of the cropping box 
                                             in term of spatial position.

    Returns:
        Tuple[ndarray, ndarray] : Two numpy arrays of the cropped image and roi
    """
    assert np.sum(np.array(crop_shape) % 2) == 0, "All elements of crop_shape should be even number."

    radius = [int(x / 2) - 1 for x in crop_shape]
    if center is None:
        center = list(np.array(list(center_of_mass(roi_data))).astype(int))

    center_min = np.floor(center).astype(int)
    center_max = np.ceil(center).astype(int)

    # If center_max and center_min are equal we add 1 to center_max to avoid trouble with crop.
    for i in range(3):
        center_max[i] += 1 if center_max[i] == center_min[i] else 0

    img_shape = image_data.shape

    # Pad the image and the ROI if its necessary
    padding = []
    for rad, cent_min, cent_max, shape in zip(radius, center_min, center_max, img_shape):
        padding.append(
            [abs(min(cent_min - rad, 0)), max(cent_max + rad + 1 - shape, 0)]
        )

    image_data = np.pad(image_data, tuple([tuple(x) for x in padding]))
    roi_data = np.pad(roi_data, tuple([tuple(x) for x in padding]))

    center_min = [center_min[i] + padding[i][0] for i in range(3)]
    center_max = [center_max[i] + padding[i][0] for i in range(3)]

    # Crop the image
    image_data = image_data[center_min[0] - radius[0]:center_max[0] + radius[0] + 1,
                center_min[1] - radius[1]:center_max[1] + radius[1] + 1,
                center_min[2] - radius[2]:center_max[2] + radius[2] + 1]
    roi_data = roi_data[center_min[0] - radius[0]:center_max[0] + radius[0] + 1,
                center_min[1] - radius[1]:center_max[1] + radius[1] + 1,
                center_min[2] - radius[2]:center_max[2] + radius[2] + 1]
    
    return image_data, roi_data

def compute_box(vol: np.ndarray,
                roi: np.ndarray,
                spatial_ref: imref3d,
                box_string: str) -> Tuple[np.ndarray,
                                          np.ndarray,
                                          imref3d]:
    """Computes a new box around the ROI (Region of interest) from the original box
    and updates the volume and the ``spatial_ref``.

    Args:
        vol (ndarray): ROI mask with values of 0 and 1.
        roi (ndarray): ROI mask with values of 0 and 1.
        spatial_ref (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        box_string (str): Specifies the new box to be computed

            * 'full': full imaging data as output.
            * 'box': computes the smallest bounding box.
            * Ex: 'box10' means 10 voxels in all three dimensions are added to the smallest bounding box. The number \
                after 'box' defines the number of voxels to add.
            * Ex: '2box' computes the smallest box and outputs double its \
                size. The number before 'box' defines the multiplication in size.
    
    Returns: 
        3-element tuple containing

        - ndarray: 3D array of imaging data defining the smallest box containing the ROI.
        - ndarray: 3D array of 1's and 0's defining the ROI in ROIbox.
        - imref3d: The associated imref3d object imaging data.

    Todo:
        * I would not recommend parsing different settings into a string. \
            Provide two or more parameters instead, and use None if one or more \
            are not used.
        * There is no else statement, so "new_spatial_ref" might be unset
    """

    if "box" in box_string:
        comp = box_string == "box"
        box_bound = compute_bounding_box(mask=roi)
        if not comp:
            # Always returns the first appearance
            ind_box = box_string.find("box")
            # Addition of a certain number of voxels in all dimensions
            if ind_box == 0:
                n_v = float(box_string[(ind_box+3):])
                n_v = np.array([n_v, n_v, n_v]).astype(int)
            else:  # Multiplication of the size of the box
                factor = float(box_string[0:ind_box])
                size_box = np.diff(box_bound, axis=1) + 1
                new_box = size_box * factor
                n_v = np.round((new_box - size_box)/2.0).astype(int)

            o_k = False

            while not o_k:
                border = np.zeros([3, 2])
                border[0, 0] = box_bound[0, 0] - n_v[0]
                border[0, 1] = box_bound[0, 1] + n_v[0]
                border[1, 0] = box_bound[1, 0] - n_v[1]
                border[1, 1] = box_bound[1, 1] + n_v[1]
                border[2, 0] = box_bound[2, 0] - n_v[2]
                border[2, 1] = box_bound[2, 1] + n_v[2]
                border = border + 1
                check1 = np.sum(border[:, 0] > 0)
                check2 = border[0, 1] <= vol.shape[0]
                check3 = border[1, 1] <= vol.shape[1]
                check4 = border[2, 1] <= vol.shape[2]

                check = check1 + check2 + check3 + check4

                if check == 6:
                    o_k = True
                else:
                    n_v = np.floor(n_v / 2.0)
                    if np.sum(n_v) == 0.0:
                        o_k = True
                        n_v = [0.0, 0.0, 0.0]
        else:
            # Will compute the smallest bounding box possible
            n_v = [0.0, 0.0, 0.0]

        box_bound[0, 0] -= n_v[0]
        box_bound[0, 1] += n_v[0]
        box_bound[1, 0] -= n_v[1]
        box_bound[1, 1] += n_v[1]
        box_bound[2, 0] -= n_v[2]
        box_bound[2, 1] += n_v[2]

        box_bound = box_bound.astype(int)

        vol = vol[box_bound[0, 0]:box_bound[0, 1] + 1,
                  box_bound[1, 0]:box_bound[1, 1] + 1,
                  box_bound[2, 0]:box_bound[2, 1] + 1]
        roi = roi[box_bound[0, 0]:box_bound[0, 1] + 1,
                  box_bound[1, 0]:box_bound[1, 1] + 1,
                  box_bound[2, 0]:box_bound[2, 1] + 1]

        # Resolution in mm, nothing has changed here in terms of resolution;
        # XYZ format here.
        res = np.array([spatial_ref.PixelExtentInWorldX,
                        spatial_ref.PixelExtentInWorldY,
                        spatial_ref.PixelExtentInWorldZ])

        # IJK, as required by imref3d
        size_box = (np.diff(box_bound, axis=1) + 1).tolist()
        size_box[0] = size_box[0][0]
        size_box[1] = size_box[1][0]
        size_box[2] = size_box[2][0]
        x_limit, y_limit, z_limit = intrinsicToWorld(spatial_ref, 
                                                box_bound[0, 0],
                                                box_bound[1, 0],
                                                box_bound[2, 0])
        new_spatial_ref = imref3d(size_box, res[0], res[1], res[2])

        # The limit is defined as the border of the first pixel
        new_spatial_ref.XWorldLimits = new_spatial_ref.XWorldLimits - (
            new_spatial_ref.XWorldLimits[0] - (x_limit - res[0]/2))
        new_spatial_ref.YWorldLimits = new_spatial_ref.YWorldLimits - (
            new_spatial_ref.YWorldLimits[0] - (y_limit - res[1]/2))
        new_spatial_ref.ZWorldLimits = new_spatial_ref.ZWorldLimits - (
            new_spatial_ref.ZWorldLimits[0] - (z_limit - res[2]/2))

    elif "full" in box_string:
        new_spatial_ref = spatial_ref

    return vol, roi, new_spatial_ref

def compute_bounding_box(mask:np.ndarray) -> np.ndarray:
    """Computes the indexes of the ROI (Region of interest) enclosing box 
    in all dimensions.

    Args:
        mask (ndarray): ROI mask with values of 0 and 1.

    Returns:
        ndarray: An array containing the indexes of the bounding box.
    """

    indices = np.where(np.reshape(mask, np.size(mask), order='F') == 1)
    iv, jv, kv = np.unravel_index(indices, np.shape(mask), order='F')
    box_bound = np.zeros((3, 2))
    box_bound[0, 0] = np.min(iv)
    box_bound[0, 1] = np.max(iv)
    box_bound[1, 0] = np.min(jv)
    box_bound[1, 1] = np.max(jv)
    box_bound[2, 0] = np.min(kv)
    box_bound[2, 1] = np.max(kv)

    return box_bound.astype(int)

def get_roi(medscan: MEDscan,
            name_roi: str,
            box_string: str,
            interp=False) -> Union[image_volume_obj,
                                   image_volume_obj]:
    """Computes the ROI box (box containing the region of interest)
    and associated mask from MEDscan object.

    Args:
        medscan (MEDscan): The MEDscan class object.
        name_roi (str): name of the ROI since the a volume can have multuiple ROIs.
        box_string (str): Specifies the size if the box containing the ROI

                          - 'full': full imaging data as output.
                          - 'box': computes the smallest bounding box.
                          - Ex: 'box10': 10 voxels in all three dimensions are added to \
                            the smallest bounding box. The number after 'box' defines the \
                            number of voxels to add.
                          - Ex: '2box': Computes the smallest box and outputs double its \
                            size. The number before 'box' defines the multiplication in size.

        interp (bool): True if we need to use an interpolation for box computation.

    Returns:
        2-element tuple containing

        - image_volume_obj: 3D array of imaging data defining box containing the ROI. \
            vol.data is the 3D array, vol.spatialRef is its associated imref3d object.
        - image_volume_obj: 3D array of 1's and 0's defining the ROI. \
            roi.data is the 3D array, roi.spatialRef is its associated imref3d object.
    """
    # PARSING OF ARGUMENTS
    try:
        name_structure_set = []
        delimiters = ["\+", "\-"]
        n_contour_data = len(medscan.data.ROI.indexes)

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
                name_temp = medscan.data.ROI.get_roi_name(key=j)
                if name_temp == name_roi[i]:
                    if name_structure_set:
                        # FOR DICOM + RTSTRUCT
                        name_set_temp = medscan.data.ROI.get_name_set(key=j)
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
        if medscan.type not in ["PTscan", "CTscan", "MRscan", "ADCscan"]:
            raise ValueError("Unknown scan type.")

        spatial_ref = medscan.data.volume.spatialRef
        vol = medscan.data.volume.array.astype(np.float32)

        # COMPUTING ALL MASKS
        for c in np.arange(start=0, stop=n_contour):
            contour = contour_number[c]
            # GETTING THE XYZ POINTS FROM medscan
            roi_xyz = medscan.data.ROI.get_indexes(key=contour).copy()

            # APPLYING ROTATION TO XYZ POINTS (if necessary --> MRscan)
            if hasattr(medscan.data.volume, 'scan_rot') and medscan.data.volume.scan_rot is not None:
                roi_xyz[:, [0, 1, 2]] = np.transpose(
                    medscan.data.volume.scan_rot @ np.transpose(roi_xyz[:, [0, 1, 2]]))

            # APPLYING TRANSLATION IF SIMULATION STRUCTURE AS INPUT
            # (software STAMP utility)
            if hasattr(medscan.data.volume, 'transScanToModel'):
                translation = medscan.data.volume.transScanToModel
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
                                        orientation=medscan.data.orientation,
                                        scan_type=medscan.type,
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

        medscan.radiomics.image.update(
            {('scale'+(str(medscan.params.process.scale_non_text[0])).replace('.', 'dot')): 'ERROR_PROCESSING'})

    return vol_obj, roi_obj

def compute_roi(roi_xyz: np.ndarray,
                spatial_ref: imref3d,
                orientation: str,
                scan_type: str,
                interp=False) -> np.ndarray:
    """Computes the ROI (Region of interest) mask using the XYZ coordinates.

    Args:
        roi_xyz (ndarray): array of (x,y,z) triplets defining a contour in the Patient-Based
                           Coordinate System extracted from DICOM RTstruct.
        spatial_ref (imref3d): imref3d object (same functionality of MATLAB imref3d class).
        orientation (str): Imaging data ``orientation`` (axial, sagittal or coronal).
        scan_type (str): Imaging modality (MRscan, CTscan...).
        interp (bool): Specifies if we need to use an interpolation \
            process prior to :func:`get_polygon_mask()` in the slice axis direction.

            - True: Interpolation is performed in the slice axis dimensions. To be further \
                tested, thus please use with caution (True is safer).
            - False (default): No interpolation. This can definitely be safe \
                when the RTstruct has been saved specifically for the volume of \
                interest.
        
    Returns:
        ndarray: 3D array of 1's and 0's defining the ROI mask.

    Todo:
        - Using interpolation: this part needs to be further tested.
        - Consider changing to 'if statement'. Changing ``interp`` variable here will change the ``interp`` variable everywhere
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

        V = get_polygon_mask(roi_xyz, new_spatial_ref)

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
        roi_mask = get_polygon_mask(roi_xyz, spatial_ref)

    return roi_mask

def find_spacing(points: np.ndarray,
                 scan_type: str) -> float:
    """Finds the slice spacing in mm.

    Note:
        This function works for points from at least 2 slices. If only
        one slice is present, the function returns a None.

    Args:
        points (ndarray): Array of (x,y,z) triplets defining a contour in the
            Patient-Based Coordinate System extracted from DICOM RTstruct.
        scan_type (str): Imaging modality (MRscan, CTscan...) 

    Returns:
        float: Slice spacing in mm.
    """
    decim_keep = 4  # We keep at most 4 decimals to find the slice spacing.

    # Rounding to the nearest 0.1 mm, MRI is more problematic due to arbitrary
    # orientations allowed for imaging volumes.
    if scan_type == "MRscan":
        slices = np.unique(np.around(points, 1))
    else:
        slices = np.unique(np.around(points, 2))

    n_slices = len(slices)
    if n_slices == 1:
        return None

    diff = np.abs(np.diff(slices))
    diff = np.round(diff, decim_keep)
    slice_spacing, nOcc = mode(x=diff, return_counts=True)
    if np.max(nOcc) == 1:
        slice_spacing = np.mean(diff)

    return slice_spacing
