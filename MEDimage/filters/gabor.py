import math
from itertools import product
from typing import List, Union

import numpy as np

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from .utils import convolve


class Gabor():
    """
    The Gabor filter class
    """

    def __init__(
            self,
            size: int,
            sigma: float,
            lamb: float,
            gamma: float,
            theta: float,
            rot_invariance=False,
            padding="symmetric"
        ) -> None:
        """
        The constructor of the Gabor filter. Highly inspired by Ref 1.

        Args:
            size (int): An integer that represent the length along one dimension of the kernel.
            sigma (float): A positive float that represent the scale of the Gabor filter
            lamb (float): A positive float that represent the wavelength in the Gabor filter. (mm or pixel?)
            gamma (float): A positive float that represent the spacial aspect ratio
            theta (float): Angle parameter used in the rotation matrix
            rot_invariance (bool): If true, rotation invariance will be done on the kernel and the kernel
                                   will be rotate 2*pi / theta times.
            padding: The padding type that will be used to produce the convolution

        Returns:
            None
        """

        assert ((size + 1) / 2).is_integer() and size > 0, "size should be a positive odd number."
        assert sigma > 0, "sigma should be a positive float"
        assert lamb > 0, "lamb represent the wavelength, so it should be a positive float"
        assert gamma > 0, "gamma is the ellipticity of the support of the filter, so it should be a positive float"

        self.dim = 2
        self.padding = padding
        self.size = size
        self.sigma = sigma
        self.lamb = lamb
        self.gamma = gamma
        self.theta = theta
        self.rot = rot_invariance
        self.create_kernel()

    def create_kernel(self) -> List[np.ndarray]:
        """Create the kernel of the Gabor filter
    
        Returns:
            List[ndarray]: A list of numpy 2D-array that contain the kernel of the real part and
            the imaginary part respectively.
        """

        def compute_weight(position, theta):
            k_2 = position[0]*math.cos(theta) + position[1] * math.sin(theta)
            k_1 = position[1]*math.cos(theta) - position[0] * math.sin(theta)

            common = math.e**(-(k_1**2 + (self.gamma*k_2)**2)/(2*self.sigma**2))
            real = math.cos(2*math.pi*k_1/self.lamb)
            im = math.sin(2*math.pi*k_1/self.lamb)
            return common*real, common*im

        # Rotation invariance
        nb_rot = round(2*math.pi/abs(self.theta)) if self.rot else 1
        real_list = []
        im_list = []

        for i in range(1, nb_rot+1):
            # Initialize the kernel as tensor of zeros
            real_kernel = np.zeros([self.size for _ in range(2)])
            im_kernel = np.zeros([self.size for _ in range(2)])

            for k in product(range(self.size), repeat=2):
                real_kernel[k], im_kernel[k] = compute_weight(np.array(k)-int((self.size-1)/2), self.theta*i)

            real_list.extend([real_kernel])
            im_list.extend([im_kernel])

        self.kernel = np.expand_dims(
            np.concatenate((real_list, im_list), axis=0),
            axis=1
        )

    def convolve(self,
                 images: np.ndarray,
                 orthogonal_rot=False,
                 pooling_method='mean') -> np.ndarray:
        """Filter a given image using the Gabor kernel defined during the construction of this instance.

        Args:
            images (ndarray): A n-dimensional numpy array that represent the images to filter
            orthogonal_rot (bool): If true, the 3D images will be rotated over coronal, axial and sagittal axis

        Returns:
            ndarray: The filtered image as a numpy ndarray
        """

        # Swap the second axis with the last, to convert image B, W, H, D --> B, D, H, W
        image = np.swapaxes(images, 1, 3)

        result = convolve(self.dim, self.kernel, image, orthogonal_rot, self.padding)

        # Reshape to get real and imaginary response on the first axis.
        _dim = 2 if orthogonal_rot else 1
        nb_rot = int(result.shape[_dim]/2)
        result = np.stack(np.array_split(result, np.array([nb_rot]), _dim), axis=0)

        # 2D modulus response map
        result = np.linalg.norm(result, axis=0)

        # Rotation invariance.
        if pooling_method == 'mean':
            result = np.mean(result, axis=2) if orthogonal_rot else np.mean(result, axis=1)
        elif pooling_method == 'max':
            result = np.max(result, axis=2) if orthogonal_rot else np.max(result, axis=1)
        else:
            raise ValueError("Pooling method should be either 'mean' or 'max'.")

        # Aggregate orthogonal rotation
        result = np.mean(result, axis=0) if orthogonal_rot else result
            
        return np.swapaxes(result, 1, 3)

def apply_gabor(
        input_images: Union[image_volume_obj, np.ndarray],
        medscan: MEDscan = None,
        voxel_length: float = 0.0,
        sigma: float = 0.0,
        _lambda: float = 0.0,
        gamma: float = 0.0,
        theta: float = 0.0,
        rot_invariance: bool = False,
        padding: str = "symmetric",
        orthogonal_rot: bool = False,
        pooling_method: str = "mean"
    ) -> np.ndarray:
    """Apply the Gabor filter to a given imaging data.
    
    Args:
        input_images (Union[image_volume_obj, np.ndarray]): The input images to filter.
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        voxel_length (float, optional): The voxel size of the input image.
        sigma (float, optional): A positive float that represent the scale of the Gabor filter.
        _lambda (float, optional): A positive float that represent the wavelength in the Gabor filter.
        gamma (float, optional): A positive float that represent the spacial aspect ratio.
        theta (float, optional): Angle parameter used in the rotation matrix.
        rot_invariance (bool, optional): If true, rotation invariance will be done on the kernel and the kernel
            will be rotate 2*pi / theta times.
        padding (str, optional): The padding type that will be used to produce the convolution. Check options 
            here: `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.
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
        voxel_length = medscan.params.process.scale_non_text[0]
        sigma = medscan.params.filter.gabor.sigma / voxel_length
        lamb = medscan.params.filter.gabor._lambda / voxel_length
        size = 2 * int(7 * sigma + 0.5) + 1
        _filter = Gabor(size=size,
                        sigma=sigma,
                        lamb=lamb,
                        gamma=medscan.params.filter.gabor.gamma,
                        theta=-medscan.params.filter.gabor.theta,
                        rot_invariance=medscan.params.filter.gabor.rot_invariance,
                        padding=medscan.params.filter.gabor.padding
                        )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=medscan.params.filter.gabor.orthogonal_rot)
    else:
        if not (voxel_length and sigma and _lambda and gamma and theta):
            raise ValueError("Missing parameters to build the Gabor filter.")
        # Initialize filter class params & instance
        sigma = sigma / voxel_length
        lamb = _lambda / voxel_length
        size = 2 * int(7 * sigma + 0.5) + 1
        _filter = Gabor(size=size,
                        sigma=sigma,
                        lamb=lamb,
                        gamma=gamma,
                        theta=theta,
                        rot_invariance=rot_invariance,
                        padding=padding
                        )
        # Run convolution
        result = _filter.convolve(input_images, orthogonal_rot=orthogonal_rot, pooling_method=pooling_method)
    
    if spatial_ref:
        return image_volume_obj(np.squeeze(result), spatial_ref)
    else:
        return np.squeeze(result)
