import numpy as np

from ..MEDscan import MEDscan
from ..utils.image_volume_obj import image_volume_obj
from .gabor import *
from .laws import *
from .log import *
from .mean import *
try:
    from .TexturalFilter import TexturalFilter
except ImportError:
    import_failed = True
from .wavelet import *


def apply_filter(
            medscan: MEDscan,
            vol_obj: Union[image_volume_obj, np.ndarray],
            user_set_min_val: float = None,
            feature: str = None
    ) -> Union[image_volume_obj, np.ndarray]:
    """Applies mean filter on the given data

    Args:
        medscan (MEDscan): Instance of the MEDscan class that holds the filtering params
        vol_obj (image_volume_obj): Imaging data to be filtered
        user_set_min_val (float, optional): The minimum value to use for the discretization. Defaults to None.
        feature (str, optional): The feature to extract from the family. In batch extraction, all the features 
            of the family will be extracted. Defaults to None.

    Returns:
        image_volume_obj: Filtered imaging data.
    """
    filter_type = medscan.params.filter.filter_type

    if filter_type.lower() == "mean":
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
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
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
        voxel_length = medscan.params.process.scale_non_text[0]
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
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
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
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
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
        result = _filter.convolve(input, orthogonal_rot=medscan.params.filter.gabor.orthogonal_rot)

    elif filter_type.lower().startswith("wavelet"):
        # Initialize filter class instance
        input = np.expand_dims(vol_obj.data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
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
    elif filter_type.lower() == "textural":
        if not import_failed:
            # Initialize filter class instance
            _filter = TexturalFilter(
                family=medscan.params.filter.textural.family,
            )
            # Apply filter
            vol_obj = _filter(
                vol_obj,
                size=medscan.params.filter.textural.size,
                discretization=medscan.params.filter.textural.discretization,
                local=medscan.params.filter.textural.local,
                user_set_min_val=user_set_min_val,
                feature=feature
            )
    else:
        raise ValueError(
                r'Filter name should either be: "mean", "log", "laws", "gabor" or "wavelet".'
                )

    if not filter_type.lower() == "textural":
        vol_obj.data = np.squeeze(result,axis=0)

    return vol_obj
