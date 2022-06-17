#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict

import numpy as np

from ..biomarkers.utils import find_i_x, find_v_x

_logger = logging.getLogger(__name__)


def extract_all(MEDimg, vol, vol_int_re, wd=None, user_set_range=None) -> Dict:
    """Computes Intensity-volume Histogram Features.

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values ofCT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (float, optional): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the 
            intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.

    """
    try:
        if 'type' in MEDimg.Params['IVH'] and MEDimg.Params['IVH']:
            # PET example case (definite intensity units -- continuous case)
            if MEDimg.Params['IVH']['type'] == 'FBS' or MEDimg.Params['IVH']['type'] == 'FBSequal':
                range_fbs = [0, 0]
                if not MEDimg.Params['im_range']:
                    range_fbs[0] = np.nanmin(vol_int_re)
                    range_fbs[1] = np.nanmax(vol_int_re)
                else:
                    if MEDimg.Params['im_range'][0] == -np.inf:
                        range_fbs[0] = np.nanmin(vol_int_re)
                    else:
                        range_fbs[0] = MEDimg.Params['im_range'][0]
                    if MEDimg.Params['im_range'][1] == np.inf:
                        range_fbs[1] = np.nanmax(vol_int_re)
                    else:
                        range_fbs[1] = MEDimg.Params['im_range'][1]
                # In this case, wd = wb (see discretisation.m)
                range_fbs[0] = range_fbs[0] + 0.5*wd
                # In this case, wd = wb (see discretisation.m)
                range_fbs[1] = range_fbs[1] - 0.5*wd
                user_set_range = range_fbs

            else:  # MRI example case (arbitrary intensity units)
                user_set_range = None

        else:  # CT example case (definite intensity units -- discrete case)
            user_set_range = MEDimg.Params['im_range']

        # INITIALIZATION
        X = vol[~np.isnan(vol[:])]

        if (vol is not None) & (wd is not None) & (user_set_range is not None):
            if user_set_range:
                min_val = user_set_range[0]
                max_val = user_set_range[1]
            else:
                min_val = np.min(X)
                max_val = np.max(X)
        else:
            min_val = np.min(X)
            max_val = np.max(X)

        if max_val == np.inf:
            max_val = np.max(X)

        if min_val == -np.inf:
            min_val = np.min(X)

        # Vector of grey-levels.
        # Values are generated within the half-open interval [min_val,max_val+wd)
        levels = np.arange(min_val, max_val+wd, wd)
        n_g = levels.size
        n_v = X.size

        # Initialization of final structure (Dictionary) containing all features.
        int_vol_hist = {'Fivh_V10': [],
                    'Fivh_V90': [],
                    'Fivh_I10': [],
                    'Fivh_I90': [],
                    'Fivh_V10minusV90': [],
                    'Fivh_I10minusI90': [],
                    'Fivh_auc': []}

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i])/n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels))/(np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)
        int_vol_hist['Fivh_V10'] = v10

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)
        int_vol_hist['Fivh_V90'] = v90

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)
        int_vol_hist['Fivh_I10'] = i10

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)
        int_vol_hist['Fivh_I90'] = i90

        # Volume at intensity fraction difference v10-v90
        int_vol_hist['Fivh_V10minusV90'] = v10 - v90

        # Intensity at volume fraction difference i10-i90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        int_vol_hist['Fivh_I10minusI90'] = i10 - i90

        # Area under IVH curve
        int_vol_hist['Fivh_auc'] = np.trapz(fract_vol)/(n_g - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_COMPUTATION'})
        _logger.error(message)
        print(message)

    return int_vol_hist
