import os
import sys

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath('./MEDimage/'))
sys.path.append(MODULE_DIR)

from MEDimage.filter.MEDimageFilter import Gabor


def test_gabor():
    # 3D-Phantom
    phantom = np.zeros((64,64,64))
    phantom[32,32,32] = 255
    phantom = np.expand_dims(phantom, axis=0)
    # Gabor filter parameters
    volex_length = 2
    sigma = 10
    _lambda = 4
    gamma = 1/2
    theta = np.pi/4
    mode = 'constant' # zero padding
    sigma = sigma / volex_length
    _lambda = _lambda / volex_length
    size = 2 * int(7 * sigma + 0.5) + 1
    # Initialize Gabor instance 
    _filter = Gabor(size=size,
                    sigma=sigma,
                    lamb=_lambda,
                    gamma=gamma,
                    theta=-theta,
                    padding=mode
                    )
    # Run convolution
    result = _filter.convolve(phantom)

    assert round(np.max(result), 3) == 255.0
