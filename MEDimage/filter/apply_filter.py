import numpy as np
from MEDimage.MEDimage import MEDimage
from ..utils.image_volume_obj import image_volume_obj

from .MEDimageFilter import *


def apply_filter(MEDimg: MEDimage,
                 vol_obj: image_volume_obj) -> np.ndarray:
    """Applies mean filter on the given data

    Args:
        MEDimg (MEDimage): Instance of the MEDimage class that holds filtering params
        data (ndarray): Array of the imaging data

    Returns:
        ndarray: Filtered data.
    """
    filter_type = MEDimg.params.filter.filter_type
    volex_length = MEDimg.params.process.scale_non_text[0]
    input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)

    if filter_type.lower() == "mean":
        # Initialize filter class instance
        _filter = Mean(
                    ndims=MEDimg.params.filter.mean.ndims,
                    size=MEDimg.params.filter.mean.size,
                    padding=MEDimg.params.filter.mean.padding
                )
        # Run convolution
        result = _filter.convolve(input)

    elif filter_type.lower() == "log":
        # Initialize filter class params & instance
        sigma = MEDimg.params.filter.log.sigma / volex_length
        length = 2 * int(4 * MEDimg.params.filter.log.sigma + 0.5) + 1
        _filter = LaplacianOfGaussian(
                    ndims=MEDimg.params.filter.log.ndims,
                    size=length,
                    sigma=sigma,
                    padding=MEDimg.params.filter.log.padding
                )
        # Run convolution
        result = _filter.convolve(input, orthogonal_rot=MEDimg.params.filter.log.orthogonal_rot)

    elif filter_type.lower() == "laws":
        # Initialize filter class instance
        _filter = Laws(
                    config=MEDimg.params.filter.laws.config, 
                    energy_distance=MEDimg.params.filter.laws.energy_distance,
                    rot_invariance=MEDimg.params.filter.laws.rot_invariance,
                    padding=MEDimg.params.filter.laws.padding
                )
        # Run convolution
        result = _filter.convolve(
                        input, 
                        orthogonal_rot=MEDimg.params.filter.laws.orthogonal_rot,
                        energy_image=MEDimg.params.filter.laws.energy_image
                    )
        # Extract energy image
        if MEDimg.params.filter.laws.energy_image:
            result = result[1]

    elif filter_type.lower() == "gabor":
        # Initialize filter class params & instance
        sigma = MEDimg.params.filter.gabor.sigma / volex_length
        lamb = MEDimg.params.filter.gabor._lambda / volex_length
        size = 2 * int(7 * MEDimg.params.filter.gabor.sigma + 0.5) + 1
        _filter = Gabor(size=size,
                        sigma=sigma,
                        lamb=lamb,
                        gamma=MEDimg.params.filter.gabor.gamma,
                        theta=-MEDimg.params.filter.gabor.theta,
                        rot_invariance=MEDimg.params.filter.gabor.rot_invariance,
                        padding=MEDimg.params.filter.gabor.padding
                        )
        # Run convolution
        result = _filter.convolve(input, orthogonal_rot=MEDimg.params.filter.gabor.orthogonal_rot)

    elif filter_type.lower().startswith("wavelet"):
        # Initialize filter instance
        _filter = Wavelet(ndims=MEDimg.params.filter.wavelet.ndims, 
                        wavelet_name=MEDimg.params.filter.wavelet.basis_function,
                        rot_invariance=MEDimg.params.filter.wavelet.rot_invariance,
                        padding=MEDimg.params.filter.wavelet.padding
                        )
        # Run convolution
        result = _filter.convolve(
                        input, 
                        _filter=MEDimg.params.filter.wavelet.subband, 
                        level=MEDimg.params.filter.wavelet.level
                    )

    else:
        raise ValueError(
                r'Filter name should either be: "mean", "log", "laws", "gabor" or "wavelet".'
                )

    vol_obj.data = np.squeeze(result)

    return vol_obj
