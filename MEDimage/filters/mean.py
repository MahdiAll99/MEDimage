from abc import ABC
from typing import Union

import numpy as np

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from .utils import convolve


class Mean():
    """The mean filter class"""

    def __init__(
                self,
                ndims: int,
                size: int,
                padding="symmetric"):
        """The constructor of the mean filter

        Args:
            ndims (int): Number of dimension of the kernel filter
            size (int): An integer that represent the length along one dimension of the kernel.
            padding: The padding type that will be used to produce the convolution

        Returns:
            None
        """

        assert isinstance(ndims, int) and ndims > 0, "ndims should be a positive integer"
        assert ((size+1)/2).is_integer() and size > 0, "size should be a positive odd number."

        self.padding = padding
        self.dim = ndims
        self.size = int(size)
        self.create_kernel()

    def create_kernel(self):
        """This method construct the mean kernel using the parameters specified to the constructor.

        Returns:
            ndarray: The mean kernel as a numpy multidimensional array
        """

        # Initialize the kernel as tensor of zeros
        weight = 1 / np.prod(self.size ** self.dim)
        kernel = np.ones([self.size for _ in range(self.dim)]) * weight

        self.kernel = np.expand_dims(kernel, axis=(0, 1))

    def convolve(self,
                images: np.ndarray,
                orthogonal_rot: bool = False)-> np.ndarray:
        """Filter a given image using the LoG kernel defined during the construction of this instance.
    
        Args:
            images (ndarray): A n-dimensional numpy array that represent the images to filter
            orthogonal_rot (bool, optional): If true, the 3D images will be rotated over coronal, axial and sagittal axis

        Returns:
            ndarray: The filtered image
        """
        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(images, 1, 3)
        result = np.squeeze(convolve(self.dim, self.kernel, image, orthogonal_rot, self.padding), axis=1)
        return np.swapaxes(result, 1, 3)

def apply_mean(
            input_images: Union[np.ndarray, image_volume_obj],
            medscan: MEDscan = None,
            ndims: int = 3,
            size: int = 15,
            padding: str = "symmetric",
            orthogonal_rot: bool = False
            ) -> np.ndarray:
    """Apply the mean filter to the input image

    Args:
        input_images (ndarray): The images to filter.
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        ndims (int, optional): The number of dimensions of the input image.
        size (int, optional): The size of the kernel.
        padding (str, optional): The padding type that will be used to produce the convolution. 
            Check options here: `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.
        orthogonal_rot (bool, optional): If true, the 3D images will be rotated over coronal, axial and sagittal axis.

    Returns:
        ndarray: The filtered image.
    """
    # Check if the input is a numpy array or a Image volume object
    spatial_ref = None
    if type(input_images) == image_volume_obj:
        spatial_ref = input_images.spatialRef
        input_images = input_images.data
    
    # Convert to shape : (B, W, H, D)
    input_images = np.expand_dims(input_images.astype(np.float64), axis=0) 
    
    if medscan:
        # Initialize filter class instance
        _filter = Mean(
                ndims=medscan.params.filter.mean.ndims,
                size=medscan.params.filter.mean.size,
                padding=medscan.params.filter.mean.padding
                )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=medscan.params.filter.mean.orthogonal_rot)
    else:
        # Initialize filter class instance
        _filter = Mean(
                    ndims=ndims,
                    size=size,
                    padding=padding,
                )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=orthogonal_rot)

    if spatial_ref:
        return image_volume_obj(np.squeeze(result), spatial_ref)
    else:
        return np.squeeze(result)
