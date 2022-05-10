#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Sequence, Tuple, Union

import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import center_of_mass


def voxel_to_spatial(affine, voxel_pos: list) -> np.array:
    """
    Convert voxel position into spatial position.

    Args:
        affine (ndarray): Affine matrix.
        voxel_pos (list): A list that correspond to the location in voxel.

    Returns:
        ndarray: A numpy array that correspond to the spatial position in mm.

    """
    m = affine[:3, :3]
    translation = affine[:3, 3]
    return m.dot(voxel_pos) + translation

def spatial_to_voxel(self, spatial_pos: list) -> np.array:
    """
    Convert spatial position into voxel position

    Args:
        affine (ndarray): Affine matrix.
        spatial_pos (list): A list that correspond to the spatial location in mm.

    Returns:
        ndarray: A numpy array that correspond to the position in the voxel.

    """
    affine = self.get_nifti().affine
    affine = np.linalg.inv(affine)

    m = affine[:3, :3]
    translation = affine[:3, 3]
    return m.dot(spatial_pos) + translation

def cropBox(image: Nifti1Image,
        roi: Nifti1Image,
        crop_shape: List[int],
        center: Union[Sequence[int], None] = None
        ) -> Tuple[Nifti1Image, Nifti1Image]:
    """
    Crop a part of the image and the ROI and save the image and the ROI if requested.

    Args:
        image (Nifti1Image): Class for the file NIfTI1 format image that will be cropped.
        roi (Nifti1Image): Class for the file NIfTI1 format ROI that will be cropped.
        crop_shape (List[int]): The dimension of the region to crop in term of number of voxel.
        center (Union[Sequence[int], None]): A list that indicate the center of the cropping box 
            in term of spatial position.
        save (bool): A boolean that indicate if we need to save the image after this operation.
            If keep memory is false, than the image will be saved either if save is true or false.
        save_path (str): A string that indicate the path where the images will be save

    Returns:
        Tuple[Nifti1Image, Nifti1Image] : Two numpy arrays of the cropped image and roi 

    """
    assert np.sum(np.array(crop_shape) % 2) == 0, "All element of crop_shape should be even number."

    radius = [int(x / 2) - 1 for x in crop_shape]
    if center is None:
        list(np.array(list(center_of_mass(roi))).astype(int))

    center_min = np.floor(spatial_to_voxel(center)).astype(int)
    center_max = np.ceil(spatial_to_voxel(center)).astype(int)

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

    image_data = image.get_fdata()
    roi_data = roi.get_fdata()

    image_data = np.pad(image_data, tuple([tuple(x) for x in padding]))
    roi = np.pad(roi, tuple([tuple(x) for x in padding]))
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
