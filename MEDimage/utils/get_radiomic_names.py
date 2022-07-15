#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np


def get_radiomic_names(roi_names: np.ndarray, roi_type):
    """
    """

    n_names = np.size(roi_names)[0]
    radiomic_names = [0] * n_names
    for n in range(0, n_names):
        radiomic_names[n] = roi_names[n, 0]+'__'+roi_names[n, 1] + \
            '('+roi_type+').'+roi_names[n, 2]+'.npy'

    return radiomic_names
