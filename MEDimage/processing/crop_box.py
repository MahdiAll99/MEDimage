#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Sequence, Tuple, Union

import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import center_of_mass


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
