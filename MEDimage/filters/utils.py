from typing import List

import numpy as np
from scipy.signal import fftconvolve


def pad_imgs(
            images: np.ndarray,
            padding_length: List,
            axis: List,
            mode: str
            )-> np.ndarray:
    """Apply padding on a 3d images using a 2D padding pattern.

    Args:
        images (ndarray): a numpy array that represent the image.
        padding_length (List): The padding length that will apply on each side of each axe.
        axis (List): A list of axes on which the padding will be done.
        mode (str): The padding mode. Check options here: `numpy.pad 
            <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.

    Returns:
        ndarray: A numpy array that represent the padded image.
    """
    pad_tuple = ()
    j = 1

    for i in range(np.ndim(images)):
        if i in axis:
            pad_tuple += ((padding_length[-j], padding_length[-j]),)
            j += 1
        else:
            pad_tuple += ((0, 0),)

    return np.pad(images, pad_tuple, mode=mode)

def convolve(
        dim: int,
        kernel: np.ndarray,
        images: np.ndarray,
        orthogonal_rot: bool=False,
        mode: str = "symmetric"
    ) -> np.ndarray:
    """Convolve a given n-dimensional array with the kernel to generate a filtered image.

    Args:
        dim (int): The dimension of the images.
        kernel (ndarray): The kernel to use for the convolution.
        images (ndarray): A n-dimensional numpy array that represent a batch of images to filter.
        orthogonal_rot (bool, optional): If true, the 3D images will be rotated over coronal, axial and sagittal axis.
        mode (str, optional): The padding mode. Check options here: `numpy.pad 
            <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.

    Returns:
        ndarray: The filtered image.
    """

    in_size = np.shape(images)

    # We only handle 2D or 3D images.
    assert len(in_size) == 3 or len(in_size) == 4, \
        "The tensor should have the followed shape (B, H, W) or (B, D, H, W)"

    if not orthogonal_rot:
        # If we have a 2D kernel but a 3D images, we squeeze the tensor
        if dim < len(in_size) - 1:
            images = images.reshape((in_size[0] * in_size[1], in_size[2], in_size[3]))

        # We compute the padding size along each dimension
        padding = [int((kernel.shape[-1] - 1) / 2) for _ in range(dim)]
        pad_axis_list = [i for i in range(1, dim+1)]

        # We pad the images and we add the channel axis.
        padded_imgs = pad_imgs(images, padding, pad_axis_list, mode)
        new_imgs = np.expand_dims(padded_imgs, axis=1)

        # Operate the convolution
        if dim < len(in_size) - 1:
            # If we have a 2D kernel but a 3D images, we convolve slice by slice
            result_list = [fftconvolve(np.expand_dims(new_imgs[i], axis=0), kernel, mode='valid') for i in range(len(images))]
            result = np.squeeze(np.stack(result_list), axis=2)

        else :
            result = fftconvolve(new_imgs, kernel, mode='valid')

        # Reshape the data to retrieve the following format: (B, C, D, H, W)
        if dim < len(in_size) - 1:
            result = result.reshape((
                in_size[0], in_size[1], result.shape[1], in_size[2], in_size[3])
            ).transpose(0, 2, 1, 3, 4)

    # If we want orthogonal rotation
    else:
        coronal_imgs = images
        axial_imgs, sagittal_imgs = np.rot90(images, 1, (1, 2)), np.rot90(images, 1, (1, 3))
        
        result_coronal = convolve(dim, kernel, coronal_imgs, False, mode)
        result_axial = convolve(dim, kernel, axial_imgs, False, mode)
        result_sagittal = convolve(dim, kernel, sagittal_imgs, False, mode)

        # split and unflip and stack the result on a new axis
        result_axial = np.rot90(result_axial, 1, (3, 2))
        result_sagittal = np.rot90(result_sagittal, 1, (4, 2))

        result = np.stack([result_coronal, result_axial, result_sagittal])

    return result
