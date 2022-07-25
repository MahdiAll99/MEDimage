#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Dict
import numpy as np


def get_radiomic_names(roi_names: np.array,
                       roi_type: str) -> Dict:
    """Generates radiomics names using ``roi_names`` and ``roi_types``.

    Args:
        roi_names (np.array): array of the ROI names.
        roi_type(str): string of the ROI.

    Returns:
        dict: dict with the radiomic names
    """

    n_names = np.size(roi_names)[0]
    radiomic_names = [0] * n_names
    for n in range(0, n_names):
        radiomic_names[n] = roi_names[n, 0]+'__'+roi_names[n, 1] + \
            '('+roi_type+').'+roi_names[n, 2]+'.npy'

    return radiomic_names
