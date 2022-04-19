#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Union

import numpy as np
from numpy import ndarray

from processing.equalization import equalization


def discretisation(
    vol_RE, 
    discr_type, 
    nQ=None, 
    userSetMinVal=None, 
    ivh=False
    ) -> Union[ndarray, float]:
    """
    Quantisizes the image intensities inside the ROI.

    Note:
        FOR 'FBS' TYPE, IT IS ASSUMED THAT RE-SEGMENTATION WITH 
        PROPER RANGE WAS ALREADY PERFORMED.

    Args:
        vol_RE (ndarray): 3D array of the image volume that will be studied with 
            NaN value for the excluded voxels (voxels outside the ROI mask).
        discr_type (str): Discretisaion approach/type MUST BE: "FBS", "FBN", "FBSequal"
            or "FBNequal".
        nQ (float): Number of bins for FBS algorithm and bin width for FBN algorithm.
        userSetMinVal (float): Minimum of range re-segmentation for FBS discretisation,
            for FBN discretisation, this value has no importance as an argument
            and will not be used.
        ivh (bool): MUST BE SET TO True FOR IVH (Intensity-Volume histogram) FEATURES.

    Returns:
        ndarray: Same input image volume but with discretised intensities.
        float: bin width.

    """

    # AZ: NOTE: the "type" variable that appeared in the MATLAB source code
    # matches the name of a standard python function. I have therefore renamed
    # this variable "discr_type"

    # PARSING ARGUMENTS
    volQuant_RE = deepcopy(vol_RE)

    if nQ is None:
        return None

    if not isinstance(nQ, float):
        nQ = float(nQ)

    if discr_type not in ["FBS", "FBN", "FBSequal", "FBNequal"]:
        raise ValueError(
            "discr_type must either be \"FBS\", \"FBN\", \"FBSequal\" or \"FBNequal\".")

    # DISCRETISATION
    if discr_type in ["FBS", "FBSequal"]:
        if userSetMinVal is not None:
            minVal = deepcopy(userSetMinVal)
        else:
            minVal = np.nanmin(volQuant_RE)
    else:
        minVal = np.nanmin(volQuant_RE)

    maxVal = np.nanmax(volQuant_RE)

    if discr_type == "FBS":
        wb = nQ
        wd = wb
        volQuant_RE = np.floor((volQuant_RE - minVal) / wb) + 1.0
    elif discr_type == "FBN":
        wb = (maxVal - minVal) / nQ
        wd = 1.0
        volQuant_RE = np.floor(
            nQ * ((volQuant_RE - minVal)/(maxVal - minVal))) + 1.0
        volQuant_RE[volQuant_RE == np.nanmax(volQuant_RE)] = nQ
    elif discr_type == "FBSequal":
        wb = nQ
        wd = wb
        volQuant_RE = equalization(volQuant_RE)
        volQuant_RE = np.floor((volQuant_RE - minVal) / wb) + 1.0
    elif discr_type == "FBNequal":
        wb = (maxVal - minVal) / nQ
        wd = 1.0
        volQuant_RE = volQuant_RE.astype(np.float32)
        volQuant_RE = equalization(volQuant_RE)
        volQuant_RE = np.floor(
            nQ * ((volQuant_RE - minVal)/(maxVal - minVal))) + 1.0
        volQuant_RE[volQuant_RE == np.nanmax(volQuant_RE)] = nQ
    if ivh and discr_type in ["FBS", "FBSequal"]:
        volQuant_RE = minVal + (volQuant_RE - 0.5) * wb

    return volQuant_RE, wd
