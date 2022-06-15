#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import Tuple

import numpy as np

from ..processing.equalization import equalization


def discretisation(
    vol_re, 
    discr_type, 
    nQ=None, 
    user_set_min_val=None, 
    ivh=False
    ) -> Tuple[np.ndarray, float]:
    """
    Quantisizes the image intensities inside the ROI.

    Note:
        FOR 'FBS' TYPE, IT IS ASSUMED THAT RE-SEGMENTATION WITH 
        PROPER RANGE WAS ALREADY PERFORMED.

    Args:
        vol_re (ndarray): 3D array of the image volume that will be studied with 
            NaN value for the excluded voxels (voxels outside the ROI mask).
        discr_type (str): Discretisaion approach/type MUST BE: "FBS", "FBN", "FBSequal"
            or "FBNequal".
        nQ (float): Number of bins for FBS algorithm and bin width for FBN algorithm.
        user_set_min_val (float): Minimum of range re-segmentation for FBS discretisation,
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
    vol_quant_re = deepcopy(vol_re)

    if nQ is None:
        return None

    if not isinstance(nQ, float):
        nQ = float(nQ)

    if discr_type not in ["FBS", "FBN", "FBSequal", "FBNequal"]:
        raise ValueError(
            "discr_type must either be \"FBS\", \"FBN\", \"FBSequal\" or \"FBNequal\".")

    # DISCRETISATION
    if discr_type in ["FBS", "FBSequal"]:
        if user_set_min_val is not None:
            min_val = deepcopy(user_set_min_val)
        else:
            min_val = np.nanmin(vol_quant_re)
    else:
        min_val = np.nanmin(vol_quant_re)

    max_val = np.nanmax(vol_quant_re)

    if discr_type == "FBS":
        wb = nQ
        wd = wb
        vol_quant_re = np.floor((vol_quant_re - min_val) / wb) + 1.0
    elif discr_type == "FBN":
        wb = (max_val - min_val) / nQ
        wd = 1.0
        vol_quant_re = np.floor(
            nQ * ((vol_quant_re - min_val)/(max_val - min_val))) + 1.0
        vol_quant_re[vol_quant_re == np.nanmax(vol_quant_re)] = nQ
    elif discr_type == "FBSequal":
        wb = nQ
        wd = wb
        vol_quant_re = equalization(vol_quant_re)
        vol_quant_re = np.floor((vol_quant_re - min_val) / wb) + 1.0
    elif discr_type == "FBNequal":
        wb = (max_val - min_val) / nQ
        wd = 1.0
        vol_quant_re = vol_quant_re.astype(np.float32)
        vol_quant_re = equalization(vol_quant_re)
        vol_quant_re = np.floor(
            nQ * ((vol_quant_re - min_val)/(max_val - min_val))) + 1.0
        vol_quant_re[vol_quant_re == np.nanmax(vol_quant_re)] = nQ
    if ivh and discr_type in ["FBS", "FBSequal"]:
        vol_quant_re = min_val + (vol_quant_re - 0.5) * wb

    return vol_quant_re, wd
