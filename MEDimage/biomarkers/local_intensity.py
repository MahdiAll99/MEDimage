#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from numpy import ndarray

from ..biomarkers.utils import get_glob_peak, get_loc_peak


def extract_all(img_obj: ndarray,
                roi_obj: ndarray,
                res: ndarray,
                intensity_type: str,
                compute_global: bool = False) -> Dict:
    """Compute Local Intensity Features.
    This features refer to Local Intensity family in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        img_obj (ndarray): Continuous image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        intensity_type (str): Type of intensity to compute. Can be "arbitrary", "definite" or "filtered".
            Will compute features only for "definite" intensity type.
        compute_global (bool, optional): If True, will compute global intensity peak, we
            recommend you don't set it to True if not necessary in your study or analysis as it
            takes too much time for calculation. Default: False.

    Returns:
        Dict: Dict of the Local Intensity Features.

    Raises:
        ValueError: If `intensity_type` is not "arbitrary", "definite" or "filtered".
    """
    assert intensity_type in ["arbitrary", "definite", "filtered"], \
        "intensity_type must be 'arbitrary', 'definite' or 'filtered'"
    
    loc_int = {'Floc_peak_local': [], 'Floc_peak_global': []}

    if intensity_type == "definite":
        loc_int['Floc_peak_local'] = (get_loc_peak(img_obj, roi_obj, res))

    # NEEDS TO BE VECTORIZED FOR FASTER CALCULATION! OR
    # SIMPLY JUST CONVOLUTE A 3D AVERAGING FILTER!
    if compute_global:
        loc_int['Floc_peak_global'] = (get_glob_peak(img_obj,roi_obj, res))

    return loc_int

def peak_local(img_obj: ndarray,
               roi_obj: ndarray,
               res: ndarray) -> float:
    """Computes local intensity peak.
    This feature refers to "Floc_peak_local" (ID = VJGA) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        img_obj (ndarray): Continuous image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Local intensity peak.
    """
    return get_loc_peak(img_obj, roi_obj, res)

def peak_global(img_obj: ndarray,
                roi_obj: ndarray,
                res: ndarray) -> float:
    """Computes global intensity peak.
    This feature refers to "Floc_peak_global" (ID = 0F91) in 
    the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`__.

    Args:
        img_obj (ndarray): Continuous image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        float: Global intensity peak.
    """
    return get_glob_peak(img_obj, roi_obj, res)
