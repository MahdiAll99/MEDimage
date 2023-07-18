import math
from itertools import product
from typing import Union

import numpy as np

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from .utils import convolve


class LaplacianOfGaussian():
    """The Laplacian of gaussian filter class."""

    def __init__(
            self,
            ndims: int,
            size: int,
            sigma: float=0.1,
            padding: str="symmetric"):
        """The constructor of the laplacian of gaussian (LoG) filter

        Args:
            ndims (int): Number of dimension of the kernel filter
            size (int): An integer that represent the length along one dimension of the kernel.
            sigma (float): The gaussian standard deviation parameter of the laplacian of gaussian filter
            padding (str): The padding type that will be used to produce the convolution

        Returns:
            None
        """

        assert isinstance(ndims, int) and ndims > 0, "ndims should be a positive integer"
        assert ((size+1)/2).is_integer() and size > 0, "size should be a positive odd number."
        assert sigma > 0, "alpha should be a positive float."

        self.dim = ndims
        self.padding = padding
        self.size = int(size)
        self.sigma = sigma
        self.create_kernel()

    def create_kernel(self) -> np.ndarray:
        """This method construct the LoG kernel using the parameters specified to the constructor
    
        Returns:
            ndarray: The laplacian of gaussian kernel as a numpy multidimensional array
        """

        def compute_weight(position):
            distance_2 = np.sum(position**2)
            # $\frac{-1}{\sigma^2} * \frac{1}{\sqrt{2 \pi} \sigma}^D = \frac{-1}{\sqrt{D/2}{2 \pi} * \sigma^{D+2}}$
            first_part = -1/((2*math.pi)**(self.dim/2) * self.sigma**(self.dim+2))

            # $(D - \frac{||k||^2}{\sigma^2}) * e^{\frac{-||k||^2}{2 \sigma^2}}$
            second_part = (self.dim - distance_2/self.sigma**2)*math.e**(-distance_2/(2 * self.sigma**2))

            return first_part * second_part

        # Initialize the kernel as tensor of zeros
        kernel = np.zeros([self.size for _ in range(self.dim)])

        for k in product(range(self.size), repeat=self.dim):
            kernel[k] = compute_weight(np.array(k)-int((self.size-1)/2))

        kernel -= np.sum(kernel)/np.prod(kernel.shape)
        self.kernel = np.expand_dims(kernel, axis=(0, 1))

    def convolve(self,
                 images: np.ndarray,
                 orthogonal_rot=False) -> np.ndarray:
        """Filter a given image using the LoG kernel defined during the construction of this instance.

        Args:
            images (ndarray): A n-dimensional numpy array that represent the images to filter
            orthogonal_rot (bool): If true, the 3D images will be rotated over coronal, axial and sagittal axis

        Returns:
            ndarray: The filtered image
        """
        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(images, 1, 3)
        result = np.squeeze(convolve(self.dim, self.kernel, image, orthogonal_rot, self.padding), axis=1)
        return np.swapaxes(result, 1, 3)

def apply_log(
        input_images: Union[np.ndarray, image_volume_obj],
        medscan: MEDscan = None,
        ndims: int = 3,
        voxel_length: float = 0.0,
        sigma: float = 0.1,
        padding: str = "symmetric",
        orthogonal_rot: bool = False
    ) -> np.ndarray:
    """Apply the mean filter to the input image

    Args:
        input_images (ndarray): The images to filter.
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        ndims (int, optional): The number of dimensions of the input image.
        voxel_length (float, optional): The voxel size of the input image.
        sigma (float, optional): standard deviation of the Gaussian, controls the scale of the convolutional operator.
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
        # Initialize filter class params & instance
        sigma = medscan.params.filter.log.sigma / voxel_length
        length = 2 * int(4 * sigma + 0.5) + 1
        _filter = LaplacianOfGaussian(
                    ndims=medscan.params.filter.log.ndims,
                    size=length,
                    sigma=sigma,
                    padding=medscan.params.filter.log.padding
                )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=medscan.params.filter.log.orthogonal_rot)
    else:
        # Initialize filter class params & instance
        sigma = sigma / voxel_length
        length = 2 * int(4 * sigma + 0.5) + 1
        _filter = LaplacianOfGaussian(
                    ndims=ndims,
                    size=length,
                    sigma=sigma,
                    padding=padding
                )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=orthogonal_rot)

    if spatial_ref:
        return image_volume_obj(np.squeeze(result), spatial_ref)
    else:
        return np.squeeze(result)
