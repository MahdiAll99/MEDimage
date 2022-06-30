import numpy as np
from MEDimage.MEDimage import MEDimage

from ..MEDimageFilter import Mean


def apply_mean(MEDimg: MEDimage, data: np.ndarray) -> np.ndarray:
    """
    Applies mean filter on the given data

    Args:
        MEDimg (MEDimage): Instance of the MEDimage class that holds filtering params
        data (ndarray): Array of the imaging data
    
    Returns:
        ndarray: Filtered data.
    """
    input = np.expand_dims(data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)
    _filter = Mean(
            ndims=MEDimg.params.filter.mean.ndims, 
            size=MEDimg.params.filter.mean.size, 
            padding=MEDimg.params.filter.mean.padding)
    result = _filter.convolve(input)

    return np.squeeze(result)
