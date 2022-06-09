import numpy as np

from ..MEDimageFilter import Mean


def apply_mean(data, params):
    """
    Applies mean filter on the imaging data
    """
    input = np.expand_dims(data.astype(np.float64), axis=0)   # Convert to shape : (B, W, H, D)

    # check if it's the main params dict that passed
    if 'imParamFilter' in params:
        params = params['imParamFilter']

    # extract Mean filter params from the passed dict    
    if 'mean' in params:
        params = params['mean']
    else:
        raise ValueError("Invalid filter parameters, mean filter params not found")

    _filter = Mean(
            ndims=params['ndims'], size=params['size'], 
            padding=params['padding'])
    result = _filter.convolve(input)

    return np.squeeze(result)
