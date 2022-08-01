#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np


def get_patient_names(roi_names: np.ndarray) -> List[str]:
    """Generates all file names for scans using CSV data. 

    Args:
        roi_names (ndarray): Array with CSV data organized as follows 
            [[patient_id], [imaging_scan_name], [imagning_modality]]
        
    Returns:
        list[str]: List of scans files name.
    """
    n_names = np.size(roi_names[0])
    patient_names = [0] * n_names
    for n in range(0, n_names):
        patient_names[n] = roi_names[0][n]+'__'+roi_names[1][n] + \
            '.'+roi_names[2][n]+'.npy'

    return patient_names
