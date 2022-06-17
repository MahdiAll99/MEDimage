#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from ..biomarkers.utils import get_glob_peak, get_loc_peak


def local_peak(img_obj, roi_obj, res) -> float:
    """
    Computes local intensity peak
    """
    return get_loc_peak(img_obj, roi_obj, res)

def global_peak(img_obj, roi_obj, res) -> float:
    """
    Computes global intensity peak
    """
    return get_glob_peak(img_obj, roi_obj, res)

def extract_all(img_obj, roi_obj, res, intensity=None) -> Dict:
    """Compute Local Intensity Features.

    Args:
        img_obj (ndarray): Continuos image intensity distribution, with no NaNs
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
    if intensity is None:
        definite = True
    elif intensity == 'arbitrary':
        definite = False
    elif intensity == 'definite':
        definite = True
    elif intensity == 'filter':
        definite = False
    else:
        raise ValueError('Fourth argument must either be "arbitrary" or \
                         "definite" or "filter"')

    loc_int = {'Floc_peak_local': [], 'Floc_peak_global': []}

    # Local grey level peak
    if definite:
        loc_int['Floc_peak_local'] = (get_loc_peak(img_obj, roi_obj, res))

        # NEEDS TO BE VECTORIZED FOR FASTER CALCULATION! OR
        # SIMPLY JUST CONVOLUTE A 3D AVERAGING FILTER!
        # loc_int['Floc_peak_global'] = (getGlobPeak(img_obj,roi_obj,res))

    return loc_int
