#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from numpy import ndarray

from ..biomarkers.utils import get_glob_peak, get_loc_peak


def extract_all(img_obj: ndarray,
                roi_obj: ndarray,
                res: ndarray,
                intensity=None) -> Dict:
    """Compute Local Intensity Features.
    This features refer to Local Intensity family in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Args:
        img_obj (ndarray): Continuous image intensity distribution, with no NaNs
            outside the ROI.
        roi_obj (ndarray): Array of the mask defining the ROI.
        res (List[float]): [a,b,c] vector specifying the resolution of the volume in mm.
            XYZ resolution (world), or JIK resolution (intrinsic matlab).
        intensity (str, optional): If 'arbitrary', some feature will not be computed.
            If 'definite', all feature will be computed. If not present as an
            argument, all features will be computed. Here, 'filter' is the same as
            'arbitrary'.

    Returns:
        Dict: Dict of the Local Intensity Features.

    Raises:
        ValueError: If `intensity` is not "arbitrary", "definite" or "filter".
    """
    # INITIALIZATION
    if intensity is None or intensity == 'definite':
        definite = True
    elif intensity == 'arbitrary' or intensity == 'filter':
        definite = False
    else:
        raise ValueError('Fourth argument must either be "arbitrary" or \
                         "definite" or "filter" or None')

    loc_int = {'Floc_peak_local': [], 'Floc_peak_global': []}

    # Local grey level peak
    if definite:
        loc_int['Floc_peak_local'] = (get_loc_peak(img_obj, roi_obj, res))

        # NEEDS TO BE VECTORIZED FOR FASTER CALCULATION! OR
        # SIMPLY JUST CONVOLUTE A 3D AVERAGING FILTER!
        # loc_int['Floc_peak_global'] = (getGlobPeak(img_obj,roi_obj,res))

    return loc_int

def peak_local(img_obj: ndarray,
               roi_obj: ndarray,
               res: ndarray) -> float:
    """Computes local intensity peak.
    This feature refers to "Floc_peak_local" (id = VJGA) in the IBSI1 reference manual
    https://arxiv.org/abs/1612.07003 (PDF)

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
    This feature refers to "Floc_peak_global" (id = 0F91) in the IBSI1 reference manual
    https://arxiv.org/abs/1612.07003 (PDF)

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
