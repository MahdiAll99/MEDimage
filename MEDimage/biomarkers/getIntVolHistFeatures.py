#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict

import numpy as np

from ..biomarkers.utils import findIX, findVX

_logger = logging.getLogger(__name__)


def getIntVolHistFeatures(MEDimg, vol, volInt_RE, wd=None, userSetRange=None) -> Dict:
    """Computes Intensity-volume Histogram Features.

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values ofCT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        volInt_RE (ndarray): 3D volume, with NaNs outside the region of interest
        wd (float, optional): Discretisation width.
        userSetRange (ndarray, optional): 1-D array with shape (1,2) of the 
            intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.

    """
    try:
        if 'type' in MEDimg.Params['IVH'] and MEDimg.Params['IVH']:
            # PET example case (definite intensity units -- continuous case)
            if MEDimg.Params['IVH']['type'] == 'FBS' or MEDimg.Params['IVH']['type'] == 'FBSequal':
                rangeFBS = [0, 0]
                if not MEDimg.Params['im_range']:
                    rangeFBS[0] = np.nanmin(volInt_RE)
                    rangeFBS[1] = np.nanmax(volInt_RE)
                else:
                    if MEDimg.Params['im_range'][0] == -np.inf:
                        rangeFBS[0] = np.nanmin(volInt_RE)
                    else:
                        rangeFBS[0] = MEDimg.Params['im_range'][0]
                    if MEDimg.Params['im_range'][1] == np.inf:
                        rangeFBS[1] = np.nanmax(volInt_RE)
                    else:
                        rangeFBS[1] = MEDimg.Params['im_range'][1]
                # In this case, wd = wb (see discretisation.m)
                rangeFBS[0] = rangeFBS[0] + 0.5*wd
                # In this case, wd = wb (see discretisation.m)
                rangeFBS[1] = rangeFBS[1] - 0.5*wd
                userSetRange = rangeFBS

            else:  # MRI example case (arbitrary intensity units)
                userSetRange = None

        else:  # CT example case (definite intensity units -- discrete case)
            userSetRange = MEDimg.Params['im_range']

        # INITIALIZATION
        X = vol[~np.isnan(vol[:])]

        if (vol is not None) & (wd is not None) & (userSetRange is not None):
            if userSetRange:
                minVal = userSetRange[0]
                maxVal = userSetRange[1]
            else:
                minVal = np.min(X)
                maxVal = np.max(X)
        else:
            minVal = np.min(X)
            maxVal = np.max(X)

        if maxVal == np.inf:
            maxVal = np.max(X)

        if minVal == -np.inf:
            minVal = np.min(X)

        # Vector of grey-levels.
        # Values are generated within the half-open interval [minVal,maxVal+wd)
        levels = np.arange(minVal, maxVal+wd, wd)
        Ng = levels.size
        Nv = X.size

        # Initialization of final structure (Dictionary) containing all features.
        intVolHist = {'Fivh_V10': [],
                    'Fivh_V90': [],
                    'Fivh_I10': [],
                    'Fivh_I90': [],
                    'Fivh_V10minusV90': [],
                    'Fivh_I10minusI90': [],
                    'Fivh_auc': []}

        # Calculating fractional volume
        fractVol = np.zeros(Ng)
        for i in range(0, Ng):
            fractVol[i] = 1 - np.sum(X < levels[i])/Nv

        # Calculating intensity fraction
        fractInt = (levels - np.min(levels))/(np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        V10 = findVX(fractInt, fractVol, 10)
        intVolHist['Fivh_V10'] = V10

        # Volume at intensity fraction 90
        V90 = findVX(fractInt, fractVol, 90)
        intVolHist['Fivh_V90'] = V90

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        I10 = findIX(levels, fractVol, 10)
        intVolHist['Fivh_I10'] = I10

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        I90 = findIX(levels, fractVol, 90)
        intVolHist['Fivh_I90'] = I90

        # Volume at intensity fraction difference V10-V90
        intVolHist['Fivh_V10minusV90'] = V10 - V90

        # Intensity at volume fraction difference I10-I90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        intVolHist['Fivh_I10minusI90'] = I10 - I90

        # Area under IVH curve
        intVolHist['Fivh_auc'] = np.trapz(fractVol)/(Ng - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_COMPUTATION'})
        _logger.error(message)
        print(message)

    return intVolHist
