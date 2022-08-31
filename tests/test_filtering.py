import os
import sys

import numpy as np

MODULE_DIR = os.path.dirname(os.path.abspath('./MEDimage/'))
sys.path.append(MODULE_DIR)

from MEDimage.filters.gabor import apply_gabor


def test_gabor():
    # 3D-Phantom
    phantom = np.zeros((64,64,64))
    phantom[32,32,32] = 255
    # Gabor filter parameters
    volex_length = 2
    sigma = 10
    _lambda = 4
    gamma = 1/2
    theta = np.pi/4
    mode = 'constant' # zero padding
    sigma = sigma / volex_length
    _lambda = _lambda / volex_length
    # Apply Gabor 
    result = apply_gabor(
                    input_images=phantom,
                    voxel_length=volex_length,
                    sigma=sigma,
                    _lambda=_lambda,
                    gamma=gamma,
                    theta=theta,
                    padding=mode
                    )

    assert round(np.max(result), 3) == 255.0
