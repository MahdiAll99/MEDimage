#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from typing import Dict

import numpy as np

from MEDimage.MEDimage import MEDimage

from ..biomarkers.utils import find_i_x, find_v_x

_logger = logging.getLogger(__name__)


def init_ivh(MEDimg: MEDimage, 
             vol: np.ndarray, 
             vol_int_re: np.ndarray, 
             wd: int, 
             user_set_range: np.ndarray=None) -> tuple[np.ndarray, np.ndarray, np.integer, np.integer]:
    """Computes Intensity-volume Histogram Features.

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    """
    try:
        if 'type' in MEDimg.params.process.ivh and MEDimg.params.process.ivh:
            # PET example case (definite intensity units -- continuous case)
            if MEDimg.params.process.ivh['type'] == 'FBS' or MEDimg.params.process.ivh['type'] == 'FBSequal':
                range_fbs = [0, 0]
                if not MEDimg.params.process.im_range:
                    range_fbs[0] = np.nanmin(vol_int_re)
                    range_fbs[1] = np.nanmax(vol_int_re)
                else:
                    if MEDimg.params.process.im_range[0] == -np.inf:
                        range_fbs[0] = np.nanmin(vol_int_re)
                    else:
                        range_fbs[0] = MEDimg.params.process.im_range[0]
                    if MEDimg.params.process.im_range[1] == np.inf:
                        range_fbs[1] = np.nanmax(vol_int_re)
                    else:
                        range_fbs[1] = MEDimg.params.process.im_range[1]
                # In this case, wd = wb (see discretisation.m)
                range_fbs[0] = range_fbs[0] + 0.5*wd
                # In this case, wd = wb (see discretisation.m)
                range_fbs[1] = range_fbs[1] - 0.5*wd
                user_set_range = range_fbs

            else:  # MRI example case (arbitrary intensity units)
                user_set_range = None

        else:  # CT example case (definite intensity units -- discrete case)
            user_set_range = MEDimg.params.process.im_range

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
        levels = np.arange(min_val, max_val + wd, wd)
        n_g = levels.size
        n_v = X.size
    
    except Exception as e:
        message = f'PROBLEM WITH INITIALIZATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_INITIALIZATION'})
        _logger.error(message)
        print(message)

    return X, levels, n_g, n_v

def extract_all(MEDimg: MEDimage, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> Dict:
    """Computes Intensity-volume Histogram Features.
    This features refer to Intensity-volume histogram family in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        Dict: Dict of the Intensity Histogram Features.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Initialization of final structure (Dictionary) containing all features.
        int_vol_hist = {'Fivh_V10': [],
                    'Fivh_V90': [],
                    'Fivh_I10': [],
                    'Fivh_I90': [],
                    'Fivh_V10minusV90': [],
                    'Fivh_I10minusI90': [],
                    'Fivh_auc': []
                    }

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i])/n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

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
        int_vol_hist['Fivh_auc'] = np.trapz(fract_vol) / (n_g - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF INTENSITY-VOLUME HISTOGRAM FEATURES \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_COMPUTATION'})
        _logger.error(message)
        print(message)

    return int_vol_hist

def V10(MEDimg: MEDimage, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction 10 feature.
    This feature refers to "Fivh_V10" (id = BC2M) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction 10 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels))/(np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V10 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_V10'})
        _logger.error(message)
        print(message)

    return v10

def V90(MEDimg: MEDimage, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction 90 feature.
    This feature refers to "Fivh_V90" (id = BC2M) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction 90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V90 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_V90'})
        _logger.error(message)
        print(message)

    return v90

def I10(MEDimg: MEDimage, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction 10 feature.
    This feature refers to "Fivh_I10" (id = GBPN) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction 10 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I10 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_I10'})
        _logger.error(message)
        print(message)

    return i10

def I90(MEDimg: MEDimage, 
        vol: np.ndarray, 
        vol_int_re: np.ndarray, 
        wd: int, 
        user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction 90 feature.
    This feature refers to "Fivh_I90" (id = GBPN) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction 90 feature.
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I90 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_I90'})
        _logger.error(message)
        print(message)

    return i90

def V10minusV90(MEDimg: MEDimage, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """Computes Volume at intensity fraction difference v10-v90
    This feature refers to "Fivh_V10minusV90" (id = DDTU) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Volume at intensity fraction difference v10-v90
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Calculating intensity fraction
        fract_int = (levels - np.min(levels)) / (np.max(levels) - np.min(levels))

        # Volume at intensity fraction 10
        v10 = find_v_x(fract_int, fract_vol, 10)

        # Volume at intensity fraction 90
        v90 = find_v_x(fract_int, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF V10minusV90 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_V10minusV90'})
        _logger.error(message)
        print(message)

    return v10 - v90

def I10minusI90(MEDimg: MEDimage, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """Computes Intensity at volume fraction difference i10-i90
    This feature refers to "Fivh_I10minusI90" (id = CNV2) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Intensity at volume fraction difference i10-i90
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Intensity at volume fraction 10
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i10 = find_i_x(levels, fract_vol, 10)

        # Intensity at volume fraction 90
        #   For initial arbitrary intensities,
        #   we will always be discretising (1000 bins).
        #   So intensities are definite here.
        i90 = find_i_x(levels, fract_vol, 90)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF I10minusI90 FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_I10minusI90'})
        _logger.error(message)
        print(message)

    return i10 - i90

def auc(MEDimg: MEDimage, 
                vol: np.ndarray, 
                vol_int_re: np.ndarray, 
                wd: int, 
                user_set_range: np.ndarray=None) -> float:
    """Computes Area under IVH curve.
    This feature refers to "Fivh_auc" (id = 9CMM) in the IBSI1 reference manual https://arxiv.org/abs/1612.07003 (PDF)

    Note:
        For the input volume:
        - Naturally discretised volume can be kept as it is (e.g. HU values of CT scans) 
        - All other volumes with continuous intensity distribution should be
            quantized (e.g., nBins = 100), with levels = [min, ..., max]

    Args:
        MEDimg (MEDimage): MEDimage instance.
        vol (ndarray): 3D volume, QUANTIZED, with NaNs outside the region of interest
        vol_int_re (ndarray): 3D volume, with NaNs outside the region of interest
        wd (int): Discretisation width.
        user_set_range (ndarray, optional): 1-D array with shape (1,2) of the intensity range.

    Returns:
        float: Area under IVH curve
    """
    try:
        # Retrieve relevant parameters from init_ivh() method.
        X, levels, n_g, n_v = init_ivh(MEDimg, vol, vol_int_re, wd, user_set_range)

        # Calculating fractional volume
        fract_vol = np.zeros(n_g)
        for i in range(0, n_g):
            fract_vol[i] = 1 - np.sum(X < levels[i]) / n_v

        # Area under IVH curve
        auc = np.trapz(fract_vol) / (n_g - 1)

    except Exception as e:
        message = f'PROBLEM WITH COMPUTATION OF AUC FEATURE \n {e}'
        MEDimg.results['intVolHist_3D'][MEDimg.Params['IVHname']].update(
            {'error': 'ERROR_AUC'})
        _logger.error(message)
        print(message)

    return auc
