import numpy as np
from MEDimage.MEDscan import MEDscan
from MEDimage.utils.image_volume_obj import image_volume_obj

from .gabor import *
from .laws import *
from .log import *
from .mean import *
from .wavelet import *


def apply_filter(
            medscan: MEDscan,
            vol_obj: image_volume_obj
    ) -> image_volume_obj:
    """Applies mean filter on the given data

    Args:
        medscan (MEDscan): Instance of the MEDscan class that holds the filtering params
        vol_obj (image_volume_obj): Imaging data to be filtered

    Returns:
        image_volume_obj: Filtered imaging data.
    """
    filter_type = medscan.params.filter.filter_type
    voxel_length = medscan.params.process.scale_non_text[0]
    input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)

    if filter_type.lower() == "mean":
        # Initialize filter class instance
        _filter = Mean(
                    ndims=medscan.params.filter.mean.ndims,
                    size=medscan.params.filter.mean.size,
                    padding=medscan.params.filter.mean.padding
                )
        # Run convolution
        result = _filter.convolve(input, orthogonal_rot=medscan.params.filter.mean.orthogonal_rot)

    elif filter_type.lower() == "log":
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
        result = _filter.convolve(input, orthogonal_rot=medscan.params.filter.log.orthogonal_rot)

    elif filter_type.lower() == "laws":
        # Initialize filter class instance
        _filter = Laws(
                    config=medscan.params.filter.laws.config, 
                    energy_distance=medscan.params.filter.laws.energy_distance,
                    rot_invariance=medscan.params.filter.laws.rot_invariance,
                    padding=medscan.params.filter.laws.padding
                )
        # Run convolution
        result = _filter.convolve(
                    input, 
                    orthogonal_rot=medscan.params.filter.laws.orthogonal_rot,
                    energy_image=medscan.params.filter.laws.energy_image
                )

    elif filter_type.lower() == "gabor":
        # Initialize filter class params & instance
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
        result = _filter.convolve(input, orthogonal_rot=medscan.params.filter.gabor.orthogonal_rot)

    elif filter_type.lower().startswith("wavelet"):
        # Initialize filter class instance
        _filter = Wavelet(
                        ndims=medscan.params.filter.wavelet.ndims, 
                        wavelet_name=medscan.params.filter.wavelet.basis_function,
                        rot_invariance=medscan.params.filter.wavelet.rot_invariance,
                        padding=medscan.params.filter.wavelet.padding
                    )
        # Run convolution
        result = _filter.convolve(
                        input, 
                        _filter=medscan.params.filter.wavelet.subband, 
                        level=medscan.params.filter.wavelet.level
                    )

    else:
        raise ValueError(
                r'Filter name should either be: "mean", "log", "laws", "gabor" or "wavelet".'
                )

    vol_obj.data = np.squeeze(result,axis=0)

    return vol_obj
