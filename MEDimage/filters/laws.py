import math
from itertools import permutations, product
from typing import List, Union

import numpy as np
from scipy.signal import fftconvolve

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from .utils import convolve, pad_imgs


class Laws():
    """
    The Laws filter class
    """

    def __init__(
                self,
                config: List = None,
                energy_distance: int = 7,
                rot_invariance: bool = False,
                padding: str = "symmetric"):
        """The constructor of the Laws filter

        Args:
            config (str): A string list of every 1D filter used to create the Laws kernel. Since the outer product is
                    not commutative, we need to use a list to specify the order of the outer product. It is not
                    recommended to use filter of different size to create the Laws kernel.
            energy_distance (float): The distance that will be used to create the energy_kernel.
            rot_invariance (bool): If true, rotation invariance will be done on the kernel.
            padding (str): The padding type that will be used to produce the convolution

        Returns:
            None
        """

        ndims = len(config)

        self.config = config
        self.energy_dist = energy_distance
        self.dim = ndims
        self.padding = padding
        self.rot = rot_invariance
        self.energy_kernel = None
        self.create_kernel()
        self.__create_energy_kernel()

    @staticmethod
    def __get_filter(name,
                     pad=False) -> np.ndarray:
        """This method create a 1D filter according to the given filter name.

        Args:
            name (float): The filter name. (Such as L3, L5, E3, E5, S3, S5, W5 or R5)
            pad (bool): If true, add zero padding of length 1 each side of kernel L3, E3 and S3

        Returns:
            ndarray: A 1D filter that is needed to construct the Laws kernel.
        """

        if name == "L3":
            ker = np.array([0, 1, 2, 1, 0]) if pad else np.array([1, 2, 1])
            return 1/math.sqrt(6) * ker
        elif name == "L5":
            return 1/math.sqrt(70) * np.array([1, 4, 6, 4, 1])
        elif name == "E3":
            ker = np.array([0, -1, 0, 1, 0]) if pad else np.array([-1, 0, 1])
            return 1 / math.sqrt(2) * ker
        elif name == "E5":
            return 1 / math.sqrt(10) * np.array([-1, -2, 0, 2, 1])
        elif name == "S3":
            ker = np.array([0, -1, 2, -1, 0]) if pad else np.array([-1, 2, -1])
            return 1 / math.sqrt(6) * ker
        elif name == "S5":
            return 1 / math.sqrt(6) * np.array([-1, 0, 2, 0, -1])
        elif name == "W5":
            return 1 / math.sqrt(10) * np.array([-1, 2, 0, -2, 1])
        elif name == "R5":
            return 1 / math.sqrt(70) * np.array([1, -4, 6, -4, 1])
        else:
            raise Exception(f"{name} is not a valid filter name. "
                            "Choose between : L3, L5, E3, E5, S3, S5, W5 or R5")

    def __verify_padding_need(self) -> bool:
        """Check if we need to pad the kernels
    
        Returns: 
            bool: A boolean that indicate if a kernel is smaller than at least one other.
        """

        ker_length = np.array([int(name[-1]) for name in self.config])

        return not(ker_length.min == ker_length.max)

    def create_kernel(self) -> np.ndarray:
        """Create the Laws by computing the outer product of 1d filter specified in the config attribute.
        Kernel = config[0] X config[1] X ... X config[n]. Where X is the outer product.

        Returns:
            ndarray: A numpy multi-dimensional arrays that represent the Laws kernel.
        """

        pad = self.__verify_padding_need()
        filter_list = np.array([[self.__get_filter(name, pad) for name in self.config]])

        if self.rot:
            filter_list = np.concatenate((filter_list, np.flip(filter_list, axis=2)), axis=0)
            prod_list = [prod for prod in product(*np.swapaxes(filter_list, 0, 1))]

            perm_list = []
            for i in range(len(prod_list)):
                perm_list.extend([perm for perm in permutations(prod_list[i])])

            filter_list = np.unique(perm_list, axis=0)

        kernel_list = []
        for perm in filter_list:
            kernel = perm[0]
            shape = kernel.shape

            for i in range(1, len(perm)):
                sub_kernel = perm[i]
                shape += np.shape(sub_kernel)
                kernel = np.outer(sub_kernel, kernel).reshape(shape)
            if self.dim == 3:
                kernel_list.extend([np.expand_dims(np.flip(kernel, axis=(1, 2)), axis=0)])
            else:
                kernel_list.extend([np.expand_dims(np.flip(kernel, axis=(0, 1)), axis=0)])

        self.kernel = np.unique(kernel_list, axis=0)

    def __create_energy_kernel(self) -> np.ndarray:
        """Create the kernel that will be used to generate Laws texture energy images

        Returns:
            ndarray: A numpy multi-dimensional arrays that represent the Laws energy kernel.
        """

        # Initialize the kernel as tensor of zeros
        kernel = np.zeros([self.energy_dist*2+1 for _ in range(self.dim)])

        for k in product(range(self.energy_dist*2 + 1), repeat=self.dim):
            position = np.array(k)-self.energy_dist
            kernel[k] = 1 if np.max(abs(position)) <= self.energy_dist else 0

        self.energy_kernel = np.expand_dims(kernel/np.prod(kernel.shape), axis=(0, 1))

    def __compute_energy_image(self,
                               images: np.ndarray) -> np.ndarray:
        """Compute the Laws texture energy images as described in (Ref 1).

        Args:
            images (ndarray): A n-dimensional numpy array that represent the filtered images

        Returns:
            ndarray: A numpy multi-dimensional array of the Laws texture energy map.
        """
        # If we have a 2D kernel but a 3D images, we swap dimension channel with dimension batch.
        images = np.swapaxes(images, 0, 1)

        # absolute image intensities are used in convolution
        result = fftconvolve(np.abs(images), self.energy_kernel, mode='valid') 

        if self.dim == 2:
            return np.swapaxes(result, axis1=0, axis2=1)
        else:
            return np.squeeze(result, axis=1)

    def convolve(self,
                 images: np.ndarray,
                 orthogonal_rot=False,
                 energy_image=False):
        """Filter a given image using the Laws kernel defined during the construction of this instance.

        Args:
            images (ndarray): A n-dimensional numpy array that represent the images to filter
            orthogonal_rot (bool): If true, the 3D images will be rotated over coronal, axial and sagittal axis
            energy_image (bool): If true, return also the Laws Texture Energy Images

        Returns:
            ndarray: The filtered image
        """
        images = np.swapaxes(images, 1, 3)

        if orthogonal_rot:
            raise NotImplementedError

        result = convolve(self.dim, self.kernel, images, orthogonal_rot, self.padding)
        result = np.amax(result, axis=1) if self.dim == 2 else np.amax(result, axis=0)

        if energy_image:
            # We pad the response map
            result = np.expand_dims(result, axis=1) if self.dim == 3 else result
            ndims = len(result.shape)

            padding = [self.energy_dist for _ in range(2 * self.dim)]
            pad_axis_list = [i for i in range(ndims - self.dim, ndims)]

            response = pad_imgs(result, padding, pad_axis_list, self.padding)

            # Free memory
            del result

            # We compute the energy map and we squeeze the second dimension of the energy maps.
            energy_imgs = self.__compute_energy_image(response)

            return np.swapaxes(energy_imgs, 1, 3)
        else:
            return np.swapaxes(result, 1, 3)

def apply_laws(
        input_images: Union[np.ndarray, image_volume_obj],
        medscan: MEDscan = None,
        config: List[str] = [],
        energy_distance: int = 7,
        padding: str = "symmetric",
        rot_invariance: bool = False,
        orthogonal_rot: bool = False,
        energy_image: bool = False,
    ) -> np.ndarray:
    """Apply the mean filter to the input image

    Args:
        input_images (ndarray): The images to filter.
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        config (List[str], optional): A string list of every 1D filter used to create the Laws kernel. Since the outer product is
            not commutative, we need to use a list to specify the order of the outer product. It is not
            recommended to use filter of different size to create the Laws kernel.
        energy_distance (int, optional): The distance of the Laws energy map from the center of the image.
        padding (str, optional): The padding type that will be used to produce the convolution. Check options 
            here: `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.
        rot_invariance (bool, optional): If true, rotation invariance will be done on the kernel.
        orthogonal_rot (bool, optional): If true, the 3D images will be rotated over coronal, axial and sagittal axis.
        energy_image (bool, optional): If true, will compute and return the Laws Texture Energy Images.

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
        _filter = Laws(
                    config=medscan.params.filter.laws.config, 
                    energy_distance=medscan.params.filter.laws.energy_distance,
                    rot_invariance=medscan.params.filter.laws.rot_invariance,
                    padding=medscan.params.filter.laws.padding
                )
        # Run convolution
        result = _filter.convolve(
                    input_images, 
                    orthogonal_rot=medscan.params.filter.laws.orthogonal_rot,
                    energy_image=medscan.params.filter.laws.energy_image
                )
    elif config:
        # Initialize filter class instance
        _filter = Laws(
                    config=config, 
                    energy_distance=energy_distance,
                    rot_invariance=rot_invariance,
                    padding=padding
                )
        # Run convolution
        result = _filter.convolve(
                        input_images, 
                        orthogonal_rot=orthogonal_rot,
                        energy_image=energy_image
                    )
    else:
        raise ValueError("Either medscan or config must be provided")
    
    if spatial_ref:
        return image_volume_obj(np.squeeze(result), spatial_ref)
    else:
        return np.squeeze(result)
