#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict

from ..biomarkers.utils import getGlobPeak, getLocPeak

def get_local_peak(imgObj, roiObj, res) -> float:
    """
    Computes local intensity peak
    """
    return getLocPeak(imgObj, roiObj, res)

def get_global_peak(imgObj, roiObj, res) -> float:
    """
    Computes global intensity peak
    """
    return getGlobPeak(imgObj, roiObj, res)

def extract_all(imgObj, roiObj, res, intensity=None) -> Dict:
    """Compute Local Intensity Features.

    Args:
        imgObj (ndarray): Continuos image intensity distribution, with no NaNs
            outside the ROI.
        roiObj (ndarray): Array of the mask defining the ROI.
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

    locInt = {'Floc_peak_local': [], 'Floc_peak_global': []}

    # Local grey level peak
    if definite:
        locInt['Floc_peak_local'] = (getLocPeak(imgObj, roiObj, res))

        # NEEDS TO BE VECTORIZED FOR FASTER CALCULATION! OR
        # SIMPLY JUST CONVOLUTE A 3D AVERAGING FILTER!
        # locInt['Floc_peak_global'] = (getGlobPeak(imgObj,roiObj,res))

    return locInt
