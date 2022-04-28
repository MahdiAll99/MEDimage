#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List

import numpy as np


def getPatientNames(roiNames) -> List[str]:
    """Generates all file names for scans using CSV data. 

    Args:
        roiNames (ndarray): Array with CSV data organized as follows 
            [[PatientID], [ImagingScanName], [ImagingModality]]
        
    Returns:
        list[str]: List of scans files name.

    """
    nNames = np.size(roiNames[0])
    patientNames = [0] * nNames
    for n in range(0, nNames):
        patientNames[n] = roiNames[0][n]+'__'+roiNames[1][n] + \
            '.'+roiNames[2][n]+'.npy'

    return patientNames
