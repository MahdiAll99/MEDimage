import math
from itertools import combinations, permutations
from typing import List, Union

import numpy as np
import pywt

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj


class Wavelet():
    """
    The wavelet filter class.
    """

    def __init__(
            self,
            ndims: int,
            wavelet_name="haar",
            padding="symmetric",
            rot_invariance=False):
        """The constructor of the wavelet filter

        Args:
            ndims (int): The number of dimension of the images that will be filter as int.
            wavelet_name (str): The name of the wavelet kernel as string.
            padding (str): The padding type that will be used to produce the convolution
            rot_invariance (bool): If true, rotation invariance will be done on the images.

        Returns:
            None
        """
        self.dim = ndims
        self.padding = padding
        self.rot = rot_invariance
        self.wavelet = None
        self.kernel_length = None
        self.create_kernel(wavelet_name)

    def create_kernel(self,
                      wavelet_name: str):
        """Get the wavelet object and his kernel length.

        Args:
            wavelet_name (str): A string that represent the wavelet name that will be use to create the kernel

        Returns:
            None
        """

        self.wavelet = pywt.Wavelet(wavelet_name)
        self.kernel_length = max(self.wavelet.rec_len, self.wavelet.dec_len)

    def __unpad(self,
                images: np.ndarray,
                padding: List) -> np.ndarray:
        """Unpad a batch of images

        Args:
            images: A numpy nd-array or a list that represent the batch of padded images.
                    The shape should be (B, H, W) or (B, H, W, D)
            padding: a list of length 2*self.dim that gives the length of padding on each side of each axis.

        Returns: 
            ndarray: A numpy nd-array or a list that represent the batch of unpadded images
        """

        if self.dim == 2:
            return images[:, padding[0]:-padding[1], padding[2]:-padding[3]]
        elif self.dim == 3:
            return images[:, padding[0]:-padding[1], padding[2]:-padding[3], padding[4]:-padding[5]]
        else:
            raise NotImplementedError

    def __get_pad_length(self,
                         image_shape: List,
                         level: int) -> np.ndarray:
        """Compute the padding length needed to have a padded image where the length
        along each axis is a multiple 2^level.

        Args:
            image_shape (List): a list of integer that describe the length of the image along each axis.
            level (int): The level of the wavelet transform

        Returns: 
            ndarray: An integer list of length 2*self.dim that gives the length of padding on each side of each axis.
        """
        padding = []
        ker_length = self.kernel_length*level
        for l in image_shape:
            padded_length = math.ceil((l + 2*(ker_length-1)) / 2**level) * 2**level - l
            padding.extend([math.floor(padded_length/2), math.ceil(padded_length/2)])

        return padding

    def _pad_imgs(self,
                  images: np.ndarray,
                  padding,
                  axis: List):
        """Apply padding on a 3d images using a 2D padding pattern (special for wavelet).

        Args:
            images: a numpy array that represent the image.
            padding: The padding length that will apply on each side of each axe.
            axis: A list of axes on which the padding will be done.

        Returns: 
            ndarray: A numpy array that represent the padded image.
        """
        pad_tuple = ()
        j = 0

        for i in range(np.ndim(images)):
            if i in axis:
                pad_tuple += ((padding[j], padding[j+1]),)
                j += 2
            else:
                pad_tuple += ((0, 0),)

        return np.pad(images, pad_tuple, mode=self.padding)
        

    def convolve(self,
                 images: np.ndarray,
                 _filter="LHL",
                 level=1)-> np.ndarray:
        """Filter a given batch of images using pywavelet.

        Args:
            images (ndarray): A n-dimensional numpy array that represent the images to filter
            _filter (str): The filter to uses.
            level (int): The number of decomposition steps to perform.

        Returns:
            ndarray: The filtered image as numpy nd-array
        """

        # We pad the images
        padding = self.__get_pad_length(np.shape(images[0]), level)
        axis_list = [i for i in range(0, self.dim)]
        images = np.expand_dims(self._pad_imgs(images[0], padding, axis_list), axis=0)

        # We generate the to collect the result from pywavelet dictionary
        _index = str().join(['a' if _filter[i] == 'L' else 'd' for i in range(len(_filter))])

        if self.rot:
            result = []
            _index_list = np.unique([str().join(perm) for perm in permutations(_index, self.dim)])

            # For each images, we flip each axis.
            for image in images:
                axis_rot = [comb for j in range(self.dim+1) for comb in combinations(np.arange(self.dim), j)]
                images_rot = [np.flip(image, axis) for axis in axis_rot]

                res_rot = []
                for i in range(len(images_rot)):
                    filtered_image = pywt.swtn(images_rot[i], self.wavelet, level=level)[0]
                    res_rot.extend([np.flip(filtered_image[j], axis=axis_rot[i]) for j in _index_list])

                result.extend([np.mean(res_rot, axis=0)])
        else:
            result = []
            for i in range(len(images)):
                result.extend([pywt.swtn(images[i], self.wavelet, level=level)[level-1][_index]])

        return self.__unpad(np.array(result), padding)

def apply_wavelet(
        input_images: Union[np.ndarray, image_volume_obj],
        medscan: MEDscan = None,
        ndims: int = 3,
        wavelet_name: str = "haar",
        subband: str = "LHL",
        level: int = 1,
        padding: str = "symmetric",
        rot_invariance: bool = False
    ) -> np.ndarray:
    """Apply the mean filter to the input image
    
    Args:
        input_images (ndarray): The image to filter.
        medscan (MEDscan, optional): The MEDscan object that will provide the filter parameters.
        ndims (int, optional): The number of dimensions of the input image.
        wavelet_name (str): The name of the wavelet kernel as string.
        level (List[str], optional): The number of decompositions steps to perform.
        subband (str, optional): String of the 1D wavelet kernels ("H" for high-pass 
            filter or "L" for low-pass filter). Must have a size of ``ndims``.
        padding (str, optional): The padding type that will be used to produce the convolution. Check options 
            here: `numpy.pad <https://numpy.org/doc/stable/reference/generated/numpy.pad.html>`__.
        rot_invariance (bool, optional): If true, rotation invariance will be done on the kernel.

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
        _filter = Wavelet(
                        ndims=medscan.params.filter.wavelet.ndims, 
                        wavelet_name=medscan.params.filter.wavelet.basis_function,
                        rot_invariance=medscan.params.filter.wavelet.rot_invariance,
                        padding=medscan.params.filter.wavelet.padding
                    )
        # Run convolution
        result = _filter.convolve(
                        input_images, 
                        _filter=medscan.params.filter.wavelet.subband, 
                        level=medscan.params.filter.wavelet.level
                    )
    else:
        # Initialize filter class instance
        _filter = Wavelet(
                        ndims=ndims, 
                        wavelet_name=wavelet_name,
                        rot_invariance=rot_invariance,
                        padding=padding
                    )
        # Run convolution
        result = _filter.convolve(
                        input_images, 
                        _filter=subband, 
                        level=level
                    )
    
    if spatial_ref:
        return image_volume_obj(np.squeeze(result), spatial_ref)
    else:
        return np.squeeze(result)